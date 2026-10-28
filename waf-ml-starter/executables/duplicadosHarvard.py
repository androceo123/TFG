#!/usr/bin/env python3
"""
Extrae todas las filas de SR-BH 2020/Harvard que pertenecen a grupos de
contenido HTTP repetido y genera dos salidas:

* un CSV completo para auditoría técnica;
* un XLSX con las hojas ``Duplicados`` y ``Diccionario_auditoria``.

Este programa es de auditoría:

* no elimina ni modifica el dataset de entrada;
* no consolida filas ni aplica OR a las etiquetas;
* incluye tanto la primera aparición como las apariciones adicionales;
* conserva las columnas originales y antepone columnas ``_audit_*``;
* documenta dentro del Excel qué significa cada campo de auditoría.

La huella reproduce ``request-canonical-blake2b128-v1``, la definición
empleada en la tesis: BLAKE2b-128 sobre método, URI, protocolo, once
encabezados y cuerpo. Timestamp, IP, puertos, respuesta y etiquetas no
participan en la identidad de la petición.

El programa hace varias pasadas secuenciales para mantener acotado el uso de
memoria. La salida se escribe de forma atómica: sólo reemplaza el destino
cuando todas las verificaciones terminan correctamente.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape


PROGRAM_VERSION = "1.1.0"
FINGERPRINT_VERSION = "request-canonical-blake2b128-v1"
FINGERPRINT_PERSON = b"srbh-reqfp-v1"

HTTP_FIELDS: Tuple[str, ...] = (
    "request_http_method",
    "request_http_request",
    "request_http_protocol",
    "request_user_agent",
    "request_referer",
    "request_host",
    "request_origin",
    "request_cookie",
    "request_content_type",
    "request_accept",
    "request_accept_language",
    "request_accept_encoding",
    "request_do_not_track",
    "request_connection",
    "request_body",
)

HEADER_COLUMN_MAP: Dict[str, str] = {
    "request_user_agent": "user-agent",
    "request_referer": "referer",
    "request_host": "host",
    "request_origin": "origin",
    "request_cookie": "cookie",
    "request_content_type": "content-type",
    "request_accept": "accept",
    "request_accept_language": "accept-language",
    "request_accept_encoding": "accept-encoding",
    "request_do_not_track": "dnt",
    "request_connection": "connection",
}

AUDIT_COLUMNS: Tuple[str, ...] = (
    "_audit_source_record",
    "_audit_http_fingerprint",
    "_audit_group_size",
    "_audit_group_rank",
    "_audit_is_representative",
    "_audit_is_excess_duplicate",
    "_audit_is_empty_http_request",
    "_audit_group_first_record",
    "_audit_group_last_record",
    "_audit_raw_http_variant_count",
    "_audit_group_has_label_variation",
    "_audit_group_has_normal_attack_mix",
    "_audit_exact_row_group_size",
    "_audit_is_exact_full_row_duplicate",
)

AUDIT_FIELD_DICTIONARY: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "_audit_source_record",
        "Entero desde 1",
        "Número ordinal del registro de datos en el CSV original. El encabezado no se cuenta y no equivale necesariamente a una línea física si existen campos con saltos de línea.",
        "Permite volver al evento original y comprobarlo en la fuente.",
    ),
    (
        "_audit_http_fingerprint",
        "Texto hexadecimal, 32 caracteres",
        "Huella BLAKE2b de 128 bits calculada sobre los 15 campos HTTP de la petición.",
        "Todas las filas con la misma huella pertenecen al mismo grupo de contenido HTTP.",
    ),
    (
        "_audit_group_size",
        "Entero ≥ 2",
        "Cantidad total de filas del dataset que comparten el mismo contenido HTTP.",
        "Indica cuántas apariciones contiene el grupo, incluida la primera.",
    ),
    (
        "_audit_group_rank",
        "Entero de 1 a group_size",
        "Posición de la aparición dentro del grupo, respetando el orden original del dataset.",
        "Los rangos de cada grupo siempre deben ser 1, 2, …, group_size.",
    ),
    (
        "_audit_is_representative",
        "0 o 1",
        "Vale 1 únicamente para la primera aparición del grupo (rank = 1).",
        "Es sólo una referencia determinista; no afirma que esa fila sea más correcta.",
    ),
    (
        "_audit_is_excess_duplicate",
        "0 o 1",
        "Vale 1 para las apariciones posteriores a la primera (rank > 1).",
        "Permite contar las repeticiones adicionales sin eliminar ninguna fila.",
    ),
    (
        "_audit_is_empty_http_request",
        "0 o 1",
        "Vale 1 cuando los 15 campos HTTP usados para definir identidad están literalmente vacíos.",
        "Las dos peticiones vacías del archivo oficial se conservan y quedan señaladas.",
    ),
    (
        "_audit_group_first_record",
        "Entero desde 1",
        "Número de registro de la primera aparición del grupo en el CSV original.",
        "Ayuda a ubicar el inicio del grupo aunque sus filas estén muy separadas.",
    ),
    (
        "_audit_group_last_record",
        "Entero desde 1",
        "Número de registro de la última aparición del grupo en el CSV original.",
        "Ayuda a ubicar el final del grupo; no implica que las apariciones sean contiguas.",
    ),
    (
        "_audit_raw_http_variant_count",
        "Entero; normalmente 1",
        "Cantidad de variantes literales de los 15 campos HTTP encontradas dentro de la huella canónica.",
        "El auditor aborta antes de escribir si encuentra más de una; por eso una salida válida muestra 1.",
    ),
    (
        "_audit_group_has_label_variation",
        "0 o 1",
        "Vale 1 si las columnas NORMAL/CAPEC no tienen exactamente el mismo vector en todas las filas del grupo.",
        "Señala discrepancias de anotación; las etiquetas originales no se consolidan ni modifican.",
    ),
    (
        "_audit_group_has_normal_attack_mix",
        "0 o 1",
        "Vale 1 si dentro del grupo existe al menos una marca NORMAL y al menos una etiqueta de ataque.",
        "Identifica el conflicto más importante entre contenido idéntico y anotaciones distintas.",
    ),
    (
        "_audit_exact_row_group_size",
        "Entero ≥ 1",
        "Cantidad de veces que se repite exactamente esta fila considerando todas las columnas originales.",
        "Un valor 1 significa que el contenido HTTP se repite, pero la fila completa no.",
    ),
    (
        "_audit_is_exact_full_row_duplicate",
        "0 o 1",
        "Vale 1 cuando la fila completa aparece más de una vez en las columnas originales.",
        "Distingue una copia exacta del evento de una petición HTTP repetida con metadatos o respuesta diferentes.",
    ),
)

EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_CELL_CHARS = 32_767

LABEL_COLUMN_RE = re.compile(r"^\d{1,3}\s+-\s+.+$")
REQUEST_LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_.-]{0,31})\s+(\S+)\s+"
    r"((?:HTTP/)?\d+(?:\.\d+)?|h2c?|h3)\s*$",
    re.IGNORECASE,
)

OFFICIAL_PROFILE = {
    "filename": "data_capec_multilabel.csv",
    "size_bytes": 436_437_661,
    "md5": "173ec515308bdce5aec19cfd5b792596",
    "sha256": "9c73c90ce6564ae48b14f7179cd864d037a6a130ef69c68c1626ec5d7ce4a910",
    "input_rows": 907_815,
    "input_columns": 38,
    "empty_http_rows": 2,
    "unique_http_fingerprints": 542_991,
    "duplicate_groups": 27_624,
    "rows_in_duplicate_groups": 392_448,
    "excess_duplicate_rows": 364_824,
    "max_group_size": 453,
    "groups_with_label_variation": 288,
    "groups_with_normal_attack_mix": 284,
    "exact_duplicate_groups": 1_415,
    "rows_in_exact_duplicate_groups": 4_161,
    "exact_excess_rows": 2_746,
    "max_exact_group_size": 10,
}


@dataclass
class GroupState:
    """Metadatos acumulados para una huella repetida."""

    expected_size: int
    first_record: int
    last_record: int
    raw_signature: bytes
    first_label_signature: Tuple[str, ...]
    seen: int = 0
    label_variation: bool = False
    normal_present: bool = False
    attack_present: bool = False


def _excel_column_name(column_number: int) -> str:
    """Convierte 1 -> A, 26 -> Z, 27 -> AA."""

    if column_number < 1:
        raise ValueError("El número de columna Excel debe ser positivo")
    chars: List[str] = []
    value = column_number
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(65 + remainder))
    return "".join(reversed(chars))


def _excel_safe_text(value: object) -> Tuple[str, bool, bool]:
    """
    Prepara texto para OOXML sin convertirlo en fórmula.

    Devuelve ``(texto, truncado, controles_escapados)``. El CSV conserva el
    valor completo; el XLSX está sujeto al límite de 32.767 caracteres de
    Excel y a las restricciones de caracteres válidos de XML 1.0.
    """

    text = "" if value is None else str(value)
    escaped_controls = False
    pieces: List[str] = []
    for char in text:
        codepoint = ord(char)
        if (
            codepoint in {0x09, 0x0A, 0x0D}
            or 0x20 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        ):
            pieces.append(char)
        else:
            escaped_controls = True
            if codepoint <= 0xFF:
                pieces.append(f"\\x{codepoint:02X}")
            elif codepoint <= 0xFFFF:
                pieces.append(f"\\u{codepoint:04X}")
            else:
                pieces.append(f"\\U{codepoint:08X}")

    safe = "".join(pieces)
    marker = "\n[TRUNCADO EN XLSX; VALOR COMPLETO EN CSV]"
    truncated = len(safe) > EXCEL_MAX_CELL_CHARS
    if truncated:
        safe = safe[: EXCEL_MAX_CELL_CHARS - len(marker)] + marker
    return safe, truncated, escaped_controls


def _xlsx_text_cell(
    reference: str,
    value: object,
    *,
    style: int = 0,
) -> Tuple[str, bool, bool]:
    safe, truncated, escaped_controls = _excel_safe_text(value)
    style_attr = f' s="{style}"' if style else ""
    return (
        f'<c r="{reference}"{style_attr} t="inlineStr">'
        f'<is><t xml:space="preserve">{xml_escape(safe)}</t></is></c>',
        truncated,
        escaped_controls,
    )


def _xlsx_integer_cell(reference: str, value: int, *, style: int = 2) -> str:
    return f'<c r="{reference}" s="{style}"><v>{int(value)}</v></c>'


def _zip_info(archive_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def _zip_write_bytes(
    archive: zipfile.ZipFile,
    archive_name: str,
    payload: str | bytes,
) -> None:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    with archive.open(_zip_info(archive_name), mode="w", force_zip64=True) as handle:
        handle.write(data)


def _xlsx_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="0"/>
  <fonts count="4">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/><scheme val="minor"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="15"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="10"/><color rgb="FF17365D"/><name val="Consolas"/><family val="3"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEAF2F8"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="3">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left/><right/><top/><bottom style="thin"><color rgb="FFD9E2F3"/></bottom><diagonal/></border>
    <border><left style="thin"><color rgb="FFB4C6E7"/></left><right style="thin"><color rgb="FFB4C6E7"/></right><top style="thin"><color rgb="FFB4C6E7"/></top><bottom style="thin"><color rgb="FFB4C6E7"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="2" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="1" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="3" fillId="4" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="0"/>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>
"""


class AuditWorkbookStream:
    """Escritor XLSX OOXML en streaming, sin dependencias externas."""

    def __init__(self, path: Path, headers: Sequence[str]):
        self.path = path
        self.headers = list(headers)
        self.last_column = _excel_column_name(len(self.headers))
        self.row_count = 1
        self.truncated_cells = 0
        self.control_escaped_cells = 0
        self._closed = False
        self.archive = zipfile.ZipFile(
            path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
            compresslevel=6,
        )
        self.data_sheet: BinaryIO = self.archive.open(
            _zip_info("xl/worksheets/sheet1.xml"),
            mode="w",
            force_zip64=True,
        )
        self._start_data_sheet()

    def _write_data_xml(self, value: str) -> None:
        self.data_sheet.write(value.encode("utf-8"))

    def _column_width(self, name: str) -> float:
        audit_widths = {
            "_audit_source_record": 18,
            "_audit_http_fingerprint": 35,
            "_audit_group_size": 17,
            "_audit_group_rank": 16,
            "_audit_is_representative": 20,
            "_audit_is_excess_duplicate": 22,
            "_audit_is_empty_http_request": 23,
            "_audit_group_first_record": 22,
            "_audit_group_last_record": 21,
            "_audit_raw_http_variant_count": 24,
            "_audit_group_has_label_variation": 25,
            "_audit_group_has_normal_attack_mix": 27,
            "_audit_exact_row_group_size": 24,
            "_audit_is_exact_full_row_duplicate": 27,
        }
        if name in audit_widths:
            return float(audit_widths[name])
        if name in {"request_http_request", "request_body"}:
            return 48.0
        if name in {"request_user_agent", "request_cookie"}:
            return 55.0
        if name in {"request_referer", "request_origin"}:
            return 38.0
        if name == "timestamp":
            return 27.0
        if name in {"src_ip", "dst_ip"}:
            return 17.0
        if name.startswith("response_http_status_message"):
            return 24.0
        if LABEL_COLUMN_RE.match(name):
            return 24.0
        return 18.0

    def _start_data_sheet(self) -> None:
        columns_xml = "".join(
            f'<col min="{index}" max="{index}" width="{self._column_width(name):.1f}" customWidth="1"/>'
            for index, name in enumerate(self.headers, start=1)
        )
        header_cells: List[str] = []
        for index, name in enumerate(self.headers, start=1):
            cell, _, _ = _xlsx_text_cell(
                f"{_excel_column_name(index)}1", name, style=1
            )
            header_cells.append(cell)
        self._write_data_xml(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView tabSelected="1" workbookViewId="0">'
            '<pane xSplit="4" ySplit="1" topLeftCell="E2" activePane="bottomRight" state="frozen"/>'
            '<selection pane="bottomRight" activeCell="E2" sqref="E2"/>'
            '</sheetView></sheetViews>'
            '<sheetFormatPr defaultRowHeight="15"/>'
            f"<cols>{columns_xml}</cols>"
            '<sheetData><row r="1" ht="36" customHeight="1">'
            + "".join(header_cells)
            + "</row>"
        )

    def write_data_row(self, values: Sequence[object]) -> None:
        if len(values) != len(self.headers):
            raise ValueError(
                f"Fila XLSX inválida: {len(values)} celdas, esperadas {len(self.headers)}"
            )
        self.row_count += 1
        if self.row_count > EXCEL_MAX_ROWS:
            raise RuntimeError(
                f"La hoja Duplicados excede el límite Excel de {EXCEL_MAX_ROWS:,} filas"
            )
        cells: List[str] = []
        for index, value in enumerate(values, start=1):
            reference = f"{_excel_column_name(index)}{self.row_count}"
            if index <= len(AUDIT_COLUMNS) and index != 2:
                cells.append(_xlsx_integer_cell(reference, int(value)))
            else:
                style = 7 if index == 2 else 0
                cell, truncated, escaped_controls = _xlsx_text_cell(
                    reference, value, style=style
                )
                cells.append(cell)
                self.truncated_cells += int(truncated)
                self.control_escaped_cells += int(escaped_controls)
        self._write_data_xml(
            f'<row r="{self.row_count}">' + "".join(cells) + "</row>"
        )

    def _dictionary_sheet_xml(self) -> str:
        rows: List[str] = []

        def text_cell(row: int, column: int, value: str, style: int) -> str:
            cell, _, _ = _xlsx_text_cell(
                f"{_excel_column_name(column)}{row}", value, style=style
            )
            return cell

        rows.append(
            '<row r="1" ht="30" customHeight="1">'
            + text_cell(1, 1, "Diccionario de campos de auditoría", 3)
            + "</row>"
        )
        rows.append(
            '<row r="2" ht="48" customHeight="1">'
            + text_cell(
                2,
                1,
                "Duplicado HTTP significa igualdad del contenido de la petición en 15 campos. No significa necesariamente que las 38 columnas del evento sean idénticas.",
                4,
            )
            + "</row>"
        )
        rows.append(
            '<row r="3" ht="48" customHeight="1">'
            + text_cell(
                3,
                1,
                "Ejemplo: _audit_group_size = 3 junto con _audit_exact_row_group_size = 1 indica tres peticiones HTTP iguales, pero esta fila completa aparece una sola vez.",
                4,
            )
            + "</row>"
        )
        dictionary_headers = ("Campo", "Tipo / valores", "Significado", "Cómo interpretarlo")
        rows.append(
            '<row r="4" ht="32" customHeight="1">'
            + "".join(
                text_cell(4, index, value, 1)
                for index, value in enumerate(dictionary_headers, start=1)
            )
            + "</row>"
        )
        for row_number, definition in enumerate(AUDIT_FIELD_DICTIONARY, start=5):
            rows.append(
                f'<row r="{row_number}" ht="62" customHeight="1">'
                + text_cell(row_number, 1, definition[0], 6)
                + text_cell(row_number, 2, definition[1], 5)
                + text_cell(row_number, 3, definition[2], 5)
                + text_cell(row_number, 4, definition[3], 5)
                + "</row>"
            )

        identity_title_row = 20
        identity_note_row = 21
        excluded_title_row = 22
        excluded_note_row = 23
        safety_title_row = 24
        safety_note_row = 25
        rows.extend(
            [
                f'<row r="{identity_title_row}" ht="26" customHeight="1">'
                + text_cell(
                    identity_title_row,
                    1,
                    "Campos que definen el contenido HTTP idéntico",
                    3,
                )
                + "</row>",
                f'<row r="{identity_note_row}" ht="68" customHeight="1">'
                + text_cell(
                    identity_note_row,
                    1,
                    ", ".join(HTTP_FIELDS),
                    4,
                )
                + "</row>",
                f'<row r="{excluded_title_row}" ht="26" customHeight="1">'
                + text_cell(
                    excluded_title_row,
                    1,
                    "Campos excluidos de la huella",
                    3,
                )
                + "</row>",
                f'<row r="{excluded_note_row}" ht="46" customHeight="1">'
                + text_cell(
                    excluded_note_row,
                    1,
                    "Timestamp, IP de origen/destino, puertos, campos de respuesta y etiquetas NORMAL/CAPEC. Pueden cambiar entre filas de un mismo grupo.",
                    4,
                )
                + "</row>",
                f'<row r="{safety_title_row}" ht="26" customHeight="1">'
                + text_cell(
                    safety_title_row,
                    1,
                    "Fidelidad y seguridad del Excel",
                    3,
                )
                + "</row>",
                f'<row r="{safety_note_row}" ht="62" customHeight="1">'
                + text_cell(
                    safety_note_row,
                    1,
                    "El CSV es la evidencia completa. En el XLSX los payloads se escriben siempre como texto, nunca como fórmulas. Si una celda supera 32.767 caracteres o contiene controles no válidos en XML, el Excel muestra una representación segura y el valor íntegro permanece en el CSV.",
                    4,
                )
                + "</row>",
            ]
        )
        merge_refs = (
            "A1:D1",
            "A2:D2",
            "A3:D3",
            "A20:D20",
            "A21:D21",
            "A22:D22",
            "A23:D23",
            "A24:D24",
            "A25:D25",
        )
        merges = "".join(f'<mergeCell ref="{ref}"/>' for ref in merge_refs)
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView showGridLines="0" workbookViewId="0">'
            '<pane ySplit="4" topLeftCell="A5" activePane="bottomLeft" state="frozen"/>'
            '<selection pane="bottomLeft" activeCell="A5" sqref="A5"/>'
            '</sheetView></sheetViews>'
            '<sheetFormatPr defaultRowHeight="18"/>'
            '<cols>'
            '<col min="1" max="1" width="42" customWidth="1"/>'
            '<col min="2" max="2" width="22" customWidth="1"/>'
            '<col min="3" max="3" width="66" customWidth="1"/>'
            '<col min="4" max="4" width="66" customWidth="1"/>'
            '</cols>'
            "<sheetData>"
            + "".join(rows)
            + "</sheetData>"
            f'<autoFilter ref="A4:D{4 + len(AUDIT_FIELD_DICTIONARY)}"/>'
            f'<mergeCells count="{len(merge_refs)}">{merges}</mergeCells>'
            '<pageMargins left="0.4" right="0.4" top="0.6" bottom="0.6" header="0.2" footer="0.2"/>'
            "</worksheet>"
        )

    def finish(self) -> None:
        if self._closed:
            return
        self._write_data_xml(
            "</sheetData>"
            f'<autoFilter ref="A1:{self.last_column}{self.row_count}"/>'
            '<pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>'
            "</worksheet>"
        )
        self.data_sheet.close()

        _zip_write_bytes(
            self.archive,
            "xl/worksheets/sheet2.xml",
            self._dictionary_sheet_xml(),
        )
        _zip_write_bytes(
            self.archive,
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
""",
        )
        _zip_write_bytes(
            self.archive,
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
""",
        )
        _zip_write_bytes(
            self.archive,
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" windowHeight="12000"/></bookViews>
  <sheets>
    <sheet name="Duplicados" sheetId="1" r:id="rId1"/>
    <sheet name="Diccionario_auditoria" sheetId="2" r:id="rId2"/>
  </sheets>
  <calcPr calcId="191029" fullCalcOnLoad="1"/>
</workbook>
""",
        )
        _zip_write_bytes(
            self.archive,
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
""",
        )
        _zip_write_bytes(self.archive, "xl/styles.xml", _xlsx_styles_xml())
        self.archive.close()
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self.data_sheet.close()
        finally:
            self.archive.close()
            self._closed = True


def _set_csv_field_limit() -> None:
    """Usa el mayor límite de campo que admite la plataforma."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _open_text(path: Path, encoding: str):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding=encoding, errors="strict", newline="")
    return path.open(mode="r", encoding=encoding, errors="strict", newline="")


def _sample_text(path: Path, encoding: str, max_chars: int = 131_072) -> str:
    with _open_text(path, encoding) as handle:
        return handle.read(max_chars)


def _resolve_delimiter(path: Path, requested: str, encoding: str) -> str:
    aliases = {
        "comma": ",",
        "coma": ",",
        "tab": "\t",
        r"\t": "\t",
        "semicolon": ";",
        "puntoycoma": ";",
        "pipe": "|",
    }
    value = aliases.get(requested.lower(), requested)
    if value.lower() != "auto":
        if len(value) != 1:
            raise ValueError(
                "--sep debe ser auto, comma, tab, semicolon, pipe o un único carácter"
            )
        return value

    sample = _sample_text(path, encoding)
    if not sample:
        raise ValueError("El archivo de entrada está vacío")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else sample
        candidates = (",", "\t", ";", "|")
        counts = {candidate: first_line.count(candidate) for candidate in candidates}
        best = max(candidates, key=lambda candidate: counts[candidate])
        if counts[best] == 0:
            raise ValueError("No se pudo detectar el separador del archivo")
        return best


def _read_header(path: Path, encoding: str, delimiter: str) -> List[str]:
    with _open_text(path, encoding) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("El archivo de entrada no contiene encabezado") from exc
    if header:
        header[0] = header[0].lstrip("\ufeff")
    if len(set(header)) != len(header):
        repeated = sorted(name for name, count in Counter(header).items() if count > 1)
        raise ValueError(f"Hay nombres de columna repetidos: {repeated}")
    return header


def _iter_records(
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    expected_header: Sequence[str],
) -> Iterator[Tuple[int, List[str]]]:
    """Itera registros lógicos; el primer registro de datos tiene ordinal 1."""

    with _open_text(path, encoding) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("El archivo de entrada no contiene encabezado") from exc
        if header:
            header[0] = header[0].lstrip("\ufeff")
        if list(header) != list(expected_header):
            raise RuntimeError("El encabezado cambió entre pasadas de lectura")

        try:
            for record_number, row in enumerate(reader, start=1):
                if len(row) != len(expected_header):
                    raise ValueError(
                        "Registro CSV inválido: "
                        f"registro_lógico={record_number}, línea_física_final={reader.line_num}, "
                        f"campos={len(row)}, esperados={len(expected_header)}"
                    )
                yield record_number, row
        except csv.Error as exc:
            raise ValueError(
                f"Error CSV cerca de la línea física {reader.line_num}: {exc}"
            ) from exc


def _hash_input_file(path: Path) -> Tuple[str, str]:
    try:
        md5 = hashlib.md5(usedforsecurity=False)
    except TypeError:  # Compatibilidad con builds antiguos de Python.
        md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _original_safe_string(value: str) -> str:
    """
    Reproduce el tratamiento de nulos de la implementación de la tesis.

    La lectura de este auditor siempre entrega strings; esta función sólo se
    usa en la huella canónica. La firma literal secundaria conserva el valor
    original y hace abortar si esta normalización intentara fusionar variantes.
    """

    return "" if value.strip().lower() in {"nan", "none", "<na>", "nat"} else value


def _request_line_fields(method: str, uri: str, protocol: str) -> Tuple[str, str, str]:
    method_s = _original_safe_string(method).strip()
    uri_s = _original_safe_string(uri).strip()
    protocol_s = _original_safe_string(protocol).strip()
    match = REQUEST_LINE_RE.match(uri_s)
    if match:
        if not method_s:
            method_s = match.group(1)
        uri_s = match.group(2)
        if not protocol_s:
            protocol_s = match.group(3)
    return method_s, uri_s, protocol_s


def _canonical_protocol(protocol: str) -> str:
    raw = _original_safe_string(protocol).strip()
    low = raw.lower()
    if low in {"h2", "h2c", "http/2", "http/2.0", "2", "2.0"}:
        return "HTTP/2"
    if low in {"h3", "http/3", "http/3.0", "3", "3.0"}:
        return "HTTP/3"
    match = re.fullmatch(r"(?:http/)?(\d+)(?:\.(\d+))?", low)
    if match:
        return f"HTTP/{int(match.group(1))}.{int(match.group(2) or 0)}"
    return raw.upper()


def _update_length_prefixed(hasher, value: str) -> None:
    payload = value.encode("utf-8", errors="surrogatepass")
    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def _request_fingerprint(identity: Sequence[str]) -> bytes:
    """Devuelve los 16 bytes de la huella canónica usada en la tesis."""

    if len(identity) != len(HTTP_FIELDS):
        raise ValueError(f"Identidad HTTP inválida: {len(identity)} campos")

    values = dict(zip(HTTP_FIELDS, identity))
    method, uri, protocol = _request_line_fields(
        values["request_http_method"],
        values["request_http_request"],
        values["request_http_protocol"],
    )

    headers: Dict[str, str] = {}
    for column, header_name in HEADER_COLUMN_MAP.items():
        value = _original_safe_string(values[column])
        if value.strip():
            headers[header_name] = value

    hasher = hashlib.blake2b(digest_size=16, person=FINGERPRINT_PERSON)
    _update_length_prefixed(hasher, FINGERPRINT_VERSION)
    _update_length_prefixed(hasher, method.upper())
    _update_length_prefixed(hasher, uri)
    _update_length_prefixed(hasher, _canonical_protocol(protocol))
    for name in sorted(headers):
        _update_length_prefixed(hasher, name.lower().replace("_", "-"))
        _update_length_prefixed(hasher, headers[name].strip(" \t"))
    _update_length_prefixed(
        hasher, _original_safe_string(values["request_body"])
    )
    return hasher.digest()


def _raw_http_signature(identity: Sequence[str]) -> bytes:
    """Firma secundaria de los 15 valores literales, sin normalización."""

    hasher = hashlib.blake2b(digest_size=32, person=b"srbh-raw15-v1")
    for field_name, value in zip(HTTP_FIELDS, identity):
        _update_length_prefixed(hasher, field_name)
        _update_length_prefixed(hasher, value)
    return hasher.digest()


def _full_row_signature(row: Sequence[str], header: Sequence[str]) -> bytes:
    """Firma secundaria de todas las columnas originales."""

    hasher = hashlib.blake2b(digest_size=32, person=b"srbh-fullrow-v1")
    for field_name, value in zip(header, row):
        _update_length_prefixed(hasher, field_name)
        _update_length_prefixed(hasher, value)
    return hasher.digest()


def _identity_from_row(row: Sequence[str], indices: Sequence[int]) -> Tuple[str, ...]:
    return tuple(row[index] for index in indices)


def _label_signature(
    row: Sequence[str],
    label_indices: Sequence[int],
    label_names: Sequence[str],
    record_number: int,
) -> Tuple[str, ...]:
    values: List[str] = []
    for index, name in zip(label_indices, label_names):
        value = row[index].strip()
        if value not in {"0", "1"}:
            raise ValueError(
                f"Etiqueta no binaria en registro {record_number}, columna {name!r}: "
                f"{row[index]!r}"
            )
        values.append(value)
    return tuple(values)


def _print_progress(pass_name: str, rows: int, every: int) -> None:
    if every > 0 and rows % every == 0:
        print(f"[{pass_name}] registros procesados: {rows:,}", flush=True)


def _strict_check(actual: Dict[str, int | str], expected: Dict[str, int | str]) -> None:
    differences = {
        key: {"esperado": value, "obtenido": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if differences:
        raise RuntimeError(
            "La validación estricta contra el CSV oficial falló:\n"
            + json.dumps(differences, ensure_ascii=False, indent=2, sort_keys=True)
        )


def build_audit_csv(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    excel_output_path = (
        Path(args.excel_output).expanduser().resolve()
        if args.excel_output
        else output_path.with_suffix(".xlsx")
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")
    if input_path in {output_path, excel_output_path}:
        raise ValueError("Ninguna salida puede ser el mismo archivo que la entrada")
    if output_path == excel_output_path:
        raise ValueError("Las rutas CSV y XLSX deben ser diferentes")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"La salida CSV ya existe: {output_path}. "
            "Use --force sólo si desea reemplazarla."
        )
    if excel_output_path.exists() and not args.force:
        raise FileExistsError(
            f"La salida XLSX ya existe: {excel_output_path}. "
            "Use --force sólo si desea reemplazarla."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    excel_output_path.parent.mkdir(parents=True, exist_ok=True)
    _set_csv_field_limit()

    delimiter = _resolve_delimiter(input_path, args.sep, args.encoding)
    header = _read_header(input_path, args.encoding, delimiter)
    if any(name.startswith("_audit_") for name in header):
        raise ValueError(
            "La entrada ya contiene columnas _audit_; use el CSV crudo de Harvard"
        )
    collisions = sorted(set(header).intersection(AUDIT_COLUMNS))
    if collisions:
        raise ValueError(f"Las columnas de auditoría ya existen: {collisions}")

    missing = [field for field in HTTP_FIELDS if field not in header]
    if missing:
        raise ValueError(
            "Faltan columnas HTTP obligatorias: "
            + ", ".join(missing)
            + ". Use el CSV crudo data_capec_multilabel.csv, no sólo el parquet de features."
        )

    identity_indices = [header.index(field) for field in HTTP_FIELDS]
    label_names = [name for name in header if LABEL_COLUMN_RE.match(name)]
    label_indices = [header.index(name) for name in label_names]
    normal_label_name = "000 - Normal"
    normal_label_position = (
        label_names.index(normal_label_name) if normal_label_name in label_names else None
    )
    attack_label_positions = [
        index for index, name in enumerate(label_names) if name != normal_label_name
    ]

    file_size = input_path.stat().st_size
    md5_hex, sha256_hex = _hash_input_file(input_path)
    official_size_match = file_size == OFFICIAL_PROFILE["size_bytes"]
    official_md5_match = md5_hex == OFFICIAL_PROFILE["md5"]
    official_sha256_match = sha256_hex == OFFICIAL_PROFILE["sha256"]

    print(
        "[INPUT] "
        f"path={input_path} bytes={file_size} columns={len(header)} "
        f"sep={delimiter!r} encoding={args.encoding}",
        flush=True,
    )
    print(f"[INPUT] md5={md5_hex}", flush=True)
    print(f"[INPUT] sha256={sha256_hex}", flush=True)
    print(
        "[IDENTIDAD] "
        f"version={FINGERPRINT_VERSION} campos={len(HTTP_FIELDS)} "
        "incluye_vacias=si",
        flush=True,
    )

    if args.expected_sha256 and sha256_hex.lower() != args.expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 inesperado: esperado={args.expected_sha256.lower()} "
            f"obtenido={sha256_hex.lower()}"
        )
    if args.strict_official:
        _strict_check(
            {
                "size_bytes": file_size,
                "md5": md5_hex,
                "sha256": sha256_hex,
                "input_columns": len(header),
            },
            {
                "size_bytes": OFFICIAL_PROFILE["size_bytes"],
                "md5": OFFICIAL_PROFILE["md5"],
                "sha256": OFFICIAL_PROFILE["sha256"],
                "input_columns": OFFICIAL_PROFILE["input_columns"],
            },
        )
        if not label_names or normal_label_position is None:
            raise RuntimeError(
                "El modo --strict-official requiere las 14 columnas NORMAL/CAPEC"
            )

    # Primera pasada: cuenta huellas canónicas.
    fingerprint_counts: Counter[bytes] = Counter()
    input_rows = 0
    empty_http_rows = 0
    for record_number, row in _iter_records(
        input_path,
        encoding=args.encoding,
        delimiter=delimiter,
        expected_header=header,
    ):
        identity = _identity_from_row(row, identity_indices)
        fingerprint_counts[_request_fingerprint(identity)] += 1
        input_rows = record_number
        if all(value == "" for value in identity):
            empty_http_rows += 1
        _print_progress("PASO 1/3", input_rows, args.progress_every)

    duplicate_counts = {
        fingerprint: count
        for fingerprint, count in fingerprint_counts.items()
        if count > 1
    }
    duplicate_groups = len(duplicate_counts)
    rows_in_duplicate_groups = sum(duplicate_counts.values())
    excess_duplicate_rows = rows_in_duplicate_groups - duplicate_groups
    max_group_size = max(duplicate_counts.values(), default=0)

    # Segunda pasada: metadatos de grupo, etiquetas y duplicado de fila completa.
    group_states: Dict[bytes, GroupState] = {}
    exact_row_counts: Counter[bytes] = Counter()
    raw_variant_errors: List[Dict[str, object]] = []

    for record_number, row in _iter_records(
        input_path,
        encoding=args.encoding,
        delimiter=delimiter,
        expected_header=header,
    ):
        identity = _identity_from_row(row, identity_indices)
        fingerprint = _request_fingerprint(identity)
        expected_size = duplicate_counts.get(fingerprint)
        if expected_size is None:
            _print_progress("PASO 2/3", record_number, args.progress_every)
            continue

        raw_signature = _raw_http_signature(identity)
        if label_indices:
            labels = _label_signature(
                row, label_indices, label_names, record_number
            )
        else:
            labels = ()

        state = group_states.get(fingerprint)
        if state is None:
            state = GroupState(
                expected_size=expected_size,
                first_record=record_number,
                last_record=record_number,
                raw_signature=raw_signature,
                first_label_signature=labels,
            )
            group_states[fingerprint] = state
        else:
            state.last_record = record_number
            if raw_signature != state.raw_signature:
                if len(raw_variant_errors) < 10:
                    raw_variant_errors.append(
                        {
                            "fingerprint": fingerprint.hex(),
                            "first_record": state.first_record,
                            "different_record": record_number,
                        }
                    )
            if labels != state.first_label_signature:
                state.label_variation = True

        state.seen += 1
        if labels and normal_label_position is not None:
            state.normal_present = (
                state.normal_present or labels[normal_label_position] == "1"
            )
            state.attack_present = state.attack_present or any(
                labels[position] == "1" for position in attack_label_positions
            )

        exact_row_counts[_full_row_signature(row, header)] += 1
        _print_progress("PASO 2/3", record_number, args.progress_every)

    if raw_variant_errors:
        raise RuntimeError(
            "La huella canónica intentó agrupar variantes literales distintas de "
            "los 15 campos HTTP. Para evitar falsos duplicados no se escribió la salida. "
            "Primeros casos:\n"
            + json.dumps(raw_variant_errors, ensure_ascii=False, indent=2)
        )

    if len(group_states) != duplicate_groups:
        raise RuntimeError(
            f"Invariante rota: grupos esperados={duplicate_groups}, "
            f"grupos observados={len(group_states)}"
        )
    bad_group_sizes = [
        fingerprint.hex()
        for fingerprint, state in group_states.items()
        if state.seen != state.expected_size
    ]
    if bad_group_sizes:
        raise RuntimeError(
            "Invariante rota: tamaño observado distinto del esperado en "
            f"{len(bad_group_sizes)} grupos"
        )

    groups_with_label_variation = sum(
        int(state.label_variation) for state in group_states.values()
    )
    groups_with_normal_attack_mix = sum(
        int(state.normal_present and state.attack_present)
        for state in group_states.values()
    )
    exact_repeated = [count for count in exact_row_counts.values() if count > 1]
    exact_duplicate_groups = len(exact_repeated)
    rows_in_exact_duplicate_groups = sum(exact_repeated)
    exact_excess_rows = rows_in_exact_duplicate_groups - exact_duplicate_groups
    max_exact_group_size = max(exact_repeated, default=0)

    observed_profile: Dict[str, int | str] = {
        "input_rows": input_rows,
        "input_columns": len(header),
        "empty_http_rows": empty_http_rows,
        "unique_http_fingerprints": len(fingerprint_counts),
        "duplicate_groups": duplicate_groups,
        "rows_in_duplicate_groups": rows_in_duplicate_groups,
        "excess_duplicate_rows": excess_duplicate_rows,
        "max_group_size": max_group_size,
        "groups_with_label_variation": groups_with_label_variation,
        "groups_with_normal_attack_mix": groups_with_normal_attack_mix,
        "exact_duplicate_groups": exact_duplicate_groups,
        "rows_in_exact_duplicate_groups": rows_in_exact_duplicate_groups,
        "exact_excess_rows": exact_excess_rows,
        "max_exact_group_size": max_exact_group_size,
    }
    if args.strict_official:
        _strict_check(
            observed_profile,
            {
                key: OFFICIAL_PROFILE[key]
                for key in observed_profile
                if key in OFFICIAL_PROFILE
            },
        )
    if rows_in_duplicate_groups + 1 > EXCEL_MAX_ROWS:
        raise RuntimeError(
            "La salida supera el límite de una hoja Excel: "
            f"{rows_in_duplicate_groups:,} filas de datos + encabezado > "
            f"{EXCEL_MAX_ROWS:,}. El CSV podría generarse, pero este modo exige "
            "un XLSX completo y no dividirá silenciosamente la evidencia."
        )

    # Tercera pasada: escribe CSV y XLSX con las mismas filas, en orden fuente.
    temporary_path: Path | None = None
    temporary_excel_path: Path | None = None
    excel_writer: AuditWorkbookStream | None = None
    output_rows = 0
    representatives = 0
    excess_flags = 0
    group_ranks: Counter[bytes] = Counter()

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=excel_output_path.parent,
            prefix=f".{excel_output_path.name}.",
            suffix=".partial",
            delete=False,
        ) as excel_temp_handle:
            temporary_excel_path = Path(excel_temp_handle.name)
        excel_writer = AuditWorkbookStream(
            temporary_excel_path,
            [*AUDIT_COLUMNS, *header],
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="strict",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".partial",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            writer = csv.writer(
                output_handle,
                delimiter=",",
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n",
            )
            writer.writerow([*AUDIT_COLUMNS, *header])

            for record_number, row in _iter_records(
                input_path,
                encoding=args.encoding,
                delimiter=delimiter,
                expected_header=header,
            ):
                identity = _identity_from_row(row, identity_indices)
                fingerprint = _request_fingerprint(identity)
                group_size = duplicate_counts.get(fingerprint)
                if group_size is None:
                    _print_progress("PASO 3/3", record_number, args.progress_every)
                    continue

                group_ranks[fingerprint] += 1
                rank = group_ranks[fingerprint]
                state = group_states[fingerprint]
                exact_size = exact_row_counts[_full_row_signature(row, header)]
                is_representative = int(rank == 1)
                is_excess = int(rank > 1)

                output_values: List[object] = [
                    record_number,
                    fingerprint.hex(),
                    group_size,
                    rank,
                    is_representative,
                    is_excess,
                    int(all(value == "" for value in identity)),
                    state.first_record,
                    state.last_record,
                    1,
                    int(state.label_variation),
                    int(state.normal_present and state.attack_present),
                    exact_size,
                    int(exact_size > 1),
                    *row,
                ]
                writer.writerow(output_values)
                excel_writer.write_data_row(output_values)
                output_rows += 1
                representatives += is_representative
                excess_flags += is_excess
                _print_progress("PASO 3/3", record_number, args.progress_every)

            output_handle.flush()
            os.fsync(output_handle.fileno())
        excel_writer.finish()

        if output_rows != rows_in_duplicate_groups:
            raise RuntimeError(
                f"Invariante rota: filas de salida={output_rows}, "
                f"esperadas={rows_in_duplicate_groups}"
            )
        if representatives != duplicate_groups:
            raise RuntimeError(
                f"Invariante rota: representantes={representatives}, "
                f"grupos={duplicate_groups}"
            )
        if excess_flags != excess_duplicate_rows:
            raise RuntimeError(
                f"Invariante rota: excedentes marcados={excess_flags}, "
                f"esperados={excess_duplicate_rows}"
            )
        wrong_ranks = [
            fingerprint.hex()
            for fingerprint, count in duplicate_counts.items()
            if group_ranks[fingerprint] != count
        ]
        if wrong_ranks:
            raise RuntimeError(
                f"Invariante rota: secuencia incompleta en {len(wrong_ranks)} grupos"
            )
        if excel_writer.row_count != output_rows + 1:
            raise RuntimeError(
                f"Invariante rota: filas XLSX={excel_writer.row_count - 1}, "
                f"filas CSV={output_rows}"
            )

        required_xlsx_parts = {
            "[Content_Types].xml",
            "_rels/.rels",
            "xl/workbook.xml",
            "xl/_rels/workbook.xml.rels",
            "xl/styles.xml",
            "xl/worksheets/sheet1.xml",
            "xl/worksheets/sheet2.xml",
        }
        with zipfile.ZipFile(temporary_excel_path, mode="r") as workbook_zip:
            missing_xlsx_parts = required_xlsx_parts.difference(
                workbook_zip.namelist()
            )
            if missing_xlsx_parts:
                raise RuntimeError(
                    "XLSX incompleto; faltan partes OOXML: "
                    + ", ".join(sorted(missing_xlsx_parts))
                )
            workbook_xml = workbook_zip.read("xl/workbook.xml")
            dictionary_xml = workbook_zip.read("xl/worksheets/sheet2.xml")
            if (
                b'name="Duplicados"' not in workbook_xml
                or b'name="Diccionario_auditoria"' not in workbook_xml
                or b"_audit_exact_row_group_size" not in dictionary_xml
            ):
                raise RuntimeError(
                    "El XLSX no contiene las dos hojas o el diccionario completo"
                )

        os.replace(temporary_excel_path, excel_output_path)
        temporary_excel_path = None
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if excel_writer is not None:
            excel_writer.abort()
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if temporary_excel_path is not None and temporary_excel_path.exists():
            temporary_excel_path.unlink()

    elapsed = time.perf_counter() - started
    summary: Dict[str, object] = {
        "program_version": PROGRAM_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "fingerprint_digest_bits": 128,
        "input_path": str(input_path),
        "input_bytes": file_size,
        "input_md5": md5_hex,
        "input_sha256": sha256_hex,
        "official_size_match": official_size_match,
        "official_md5_match": official_md5_match,
        "official_sha256_match": official_sha256_match,
        "input_rows": input_rows,
        "input_columns": len(header),
        "empty_http_rows_included": empty_http_rows,
        "unique_http_fingerprints": len(fingerprint_counts),
        "duplicate_groups": duplicate_groups,
        "rows_in_duplicate_groups": rows_in_duplicate_groups,
        "representatives_in_output": representatives,
        "excess_duplicate_rows": excess_duplicate_rows,
        "max_group_size": max_group_size,
        "groups_with_label_variation": groups_with_label_variation,
        "groups_with_normal_attack_mix": groups_with_normal_attack_mix,
        "exact_duplicate_groups": exact_duplicate_groups,
        "rows_in_exact_duplicate_groups": rows_in_exact_duplicate_groups,
        "exact_excess_rows": exact_excess_rows,
        "max_exact_group_size": max_exact_group_size,
        "output_path": str(output_path),
        "excel_output_path": str(excel_output_path),
        "output_rows": output_rows,
        "excel_data_rows": output_rows,
        "excel_truncated_cells": (
            excel_writer.truncated_cells if excel_writer is not None else 0
        ),
        "excel_control_escaped_cells": (
            excel_writer.control_escaped_cells if excel_writer is not None else 0
        ),
        "strict_official": bool(args.strict_official),
        "elapsed_seconds": round(elapsed, 3),
    }

    print("[OK] CSV y XLSX de auditoría creados", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(
        "[SEGURIDAD] El CSV conserva los payloads crudos y no debe abrirse "
        "directamente como hoja de cálculo. Use el XLSX: sus payloads se guardan "
        "como texto y nunca como fórmulas.",
        flush=True,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un CSV y un XLSX con todas las filas de los grupos HTTP "
            "repetidos del dataset SR-BH 2020/Harvard. El XLSX incluye las hojas "
            "Duplicados y Diccionario_auditoria. No elimina ni consolida registros."
        )
    )
    parser.add_argument("--input", required=True, help="CSV/TSV o .gz crudo de Harvard")
    parser.add_argument(
        "--output",
        default="duplicadosHarvard/harvard_duplicados.csv",
        help=(
            "CSV de auditoría "
            "(default: duplicadosHarvard/harvard_duplicados.csv)"
        ),
    )
    parser.add_argument(
        "--excel-output",
        default="",
        help=(
            "XLSX con datos y diccionario. Si se omite, usa la ruta --output "
            "cambiando la extensión por .xlsx"
        ),
    )
    parser.add_argument(
        "--sep",
        default="auto",
        help="Separador: auto, comma, tab, semicolon, pipe o carácter literal",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Codificación de entrada (default: utf-8-sig; alternativa: latin-1)",
    )
    parser.add_argument(
        "--expected-sha256",
        default="",
        help="Si se indica, aborta cuando el SHA-256 de entrada no coincide",
    )
    parser.add_argument(
        "--strict-official",
        action="store_true",
        help="Exige hash, esquema y conteos exactos del CSV oficial auditado en la tesis",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permite reemplazar atómicamente las salidas CSV/XLSX existentes",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Frecuencia de progreso en registros; 0 desactiva (default: 100000)",
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_audit_csv(args)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
