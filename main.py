import os
import io
import json
import secrets
from pathlib import Path

import anthropic
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)

app = FastAPI(title="Argieder Smart Reports")

# Sustituye esta dirección por tu URL de GitHub Pages cuando la tengas.
ALLOWED_ORIGINS = [
    "https://calculadoradeahorro.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS_FOR_AI = 35
MAX_COLS_FOR_AI = 15


@app.get("/")
def health():
    return {"status": "ok", "app": "Argieder Smart Reports"}


def extract_document(filename: str, content: bytes):
    ext = Path(filename).suffix.lower()

    if ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="No se ha podido leer el Excel. Comprueba que no esté dañado."
            ) from exc

        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            raise HTTPException(status_code=400, detail="El Excel no contiene una tabla legible.")

        return {
            "kind": "excel",
            "columns": [str(c) for c in df.columns[:MAX_COLS_FOR_AI]],
            "rows": df.iloc[:MAX_ROWS_FOR_AI, :MAX_COLS_FOR_AI].fillna("").astype(str).values.tolist(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "dataframe": df,
        }

    if ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="No se ha podido leer el CSV.") from exc

        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            raise HTTPException(status_code=400, detail="El CSV no contiene una tabla legible.")

        return {
            "kind": "csv",
            "columns": [str(c) for c in df.columns[:MAX_COLS_FOR_AI]],
            "rows": df.iloc[:MAX_ROWS_FOR_AI, :MAX_COLS_FOR_AI].fillna("").astype(str).values.tolist(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "dataframe": df,
        }

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages[:20])
        except Exception as exc:
            raise HTTPException(status_code=400, detail="No se ha podido leer el PDF.") from exc

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Este PDF no contiene texto extraíble. Los PDF escaneados requieren OCR."
            )

        # Intenta identificar una tabla en el texto extraído.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        df = pd.DataFrame({"Contenido del PDF": lines})

        return {
            "kind": "pdf",
            "columns": ["Contenido del PDF"],
            "rows": [[line] for line in lines[:MAX_ROWS_FOR_AI]],
            "row_count": len(lines),
            "column_count": 1,
            "dataframe": df,
        }

    raise HTTPException(status_code=400, detail="Formato no admitido. Usa Excel, CSV o PDF.")


def safe_chart(df, chart_spec, index):
    """Genera gráficos con datos locales, nunca con código ejecutable generado por IA."""
    columns = [str(c) for c in df.columns]
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    chart_type = chart_spec.get("type", "bar")
    title = str(chart_spec.get("title", "Resumen"))[:100]

    if x_col not in columns or y_col not in columns:
        return None

    series = df[[x_col, y_col]].copy()
    series[y_col] = pd.to_numeric(
        series[y_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    series = series.dropna(subset=[y_col]).head(20)

    if series.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.3))
    try:
        if chart_type == "line":
            ax.plot(series[x_col].astype(str), series[y_col], marker="o")
        elif chart_type == "pie":
            series = series[series[y_col] >= 0]
            if series.empty or series[y_col].sum() <= 0:
                plt.close(fig)
                return None
            ax.pie(series[y_col], labels=series[x_col].astype(str), autopct="%1.1f%%")
        else:
            ax.bar(series[x_col].astype(str), series[y_col])

        if chart_type != "pie":
            ax.set_title(title, loc="left", fontsize=13, pad=14)
            ax.set_ylabel(str(y_col)[:60])
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.grid(axis="y", alpha=.2)
        fig.tight_layout()

        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=160, bbox_inches="tight")
        output.seek(0)
        return output
    finally:
        plt.close(fig)


def make_pdf(title, summary, findings, chart_images, source_name):
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        textColor=colors.HexColor("#102A43"),
        fontSize=23, leading=28, spaceAfter=14
    ))
    styles.add(ParagraphStyle(
        name="ReportSub", parent=styles["Normal"],
        textColor=colors.HexColor("#16817C"),
        fontSize=10, leading=15, spaceAfter=15
    ))

    story = [
        Paragraph("ARGIEDER SMART REPORTS", styles["ReportSub"]),
        Paragraph(title, styles["ReportTitle"]),
        Paragraph("Documento analizado: " + source_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Resumen ejecutivo", styles["Heading2"]),
        Paragraph(summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Puntos clave", styles["Heading2"]),
    ]

    for finding in findings:
        safe_text = str(finding).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph("• " + safe_text, styles["BodyText"]))
        story.append(Spacer(1, 5))

    for chart in chart_images:
        story.append(Spacer(1, 15))
        story.append(Image(chart, width=16.5*cm, height=8.5*cm, kind="proportional"))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Nota: informe generado automáticamente. Verifica los cálculos y el contexto antes de compartirlo con el cliente.",
        styles["Italic"]
    ))

    doc.build(story)
    output.seek(0)
    return output


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Selecciona un archivo.")

    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 20 MB.")
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="El servidor no tiene configurada la clave de IA.")

    document = extract_document(file.filename, content)
    df = document["dataframe"]

    # Los datos se envían a Anthropic para su análisis. No usar documentos
    # confidenciales hasta aprobar las condiciones de privacidad de la empresa.
    payload = {
        "filename": Path(file.filename).name[:120],
        "kind": document["kind"],
        "row_count": document["row_count"],
        "column_count": document["column_count"],
        "columns": document["columns"],
        "sample_rows": document["rows"],
    }

    prompt = """
Eres un analista de informes para clientes de una empresa de eficiencia y ahorro.
Analiza el documento y devuelve SOLO JSON válido, sin Markdown, con este formato:
{
 "title": "Título breve en español",
 "summary": "Resumen en lenguaje sencillo; no inventes datos",
 "findings": ["Conclusión basada en los datos", "..."],
 "charts": [{"type":"bar","title":"Título","x":"nombre exacto de columna","y":"nombre exacto de columna numérica"}],
 "warnings": ["Limitaciones o datos que requieren revisión"]
}
Instrucciones:
- Objetivo: a partir de este Excel, crea gráficos fáciles de entender, legibles y de muy buen aspecto para que mi cliente pueda comprender los beneficios de cada uno de los ahorros.
- Usa únicamente los datos facilitados; no inventes ahorros, porcentajes, beneficios ni cifras.
- Distingue valores observados de hipótesis.
- Recomienda como máximo 3 gráficos.
- Los tipos admitidos son bar, line y pie.
- Los campos x e y deben coincidir exactamente con las columnas facilitadas.
- Si no hay datos suficientes para un gráfico, devuelve charts como lista vacía.
- Explica claramente las limitaciones en warnings.
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest"),
            max_tokens=1800,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt + "\n\nDOCUMENTO:\n" +
                           json.dumps(payload, ensure_ascii=False)
            }],
        )
        raw = next(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("La respuesta de IA no contiene JSON.")
        analysis = json.loads(raw[start:end + 1])
    except HTTPException:
        raise
    except Exception as exc:
        # No se envían detalles internos ni secretos al navegador.
        raise HTTPException(
            status_code=502,
            detail="No se ha podido analizar el documento con la IA. Revisa la clave, el modelo y el saldo de API."
        ) from exc

    title = str(analysis.get("title", "Informe de resultados"))[:160]
    summary = str(analysis.get("summary", "No se ha podido generar un resumen."))
    findings = [str(x)[:500] for x in analysis.get("findings", [])[:8]]
    warnings = [str(x)[:500] for x in analysis.get("warnings", [])[:8]]

    chart_images = []
    for index, spec in enumerate(analysis.get("charts", [])[:3]):
        chart = safe_chart(df, spec, index)
        if chart:
            chart_images.append(chart)

    pdf = make_pdf(title, summary, findings, chart_images, Path(file.filename).name[:120])
    pdf_name = "argieder_informe_" + secrets.token_hex(4) + ".pdf"

    # El PDF se devuelve como descarga; no se guarda permanentemente en el servidor.
    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{pdf_name}"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
