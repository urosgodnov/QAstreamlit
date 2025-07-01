import streamlit as st
from pathlib import Path
import tempfile

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice


def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


def main():
    st.title("Batch Document to Markdown")

    uploaded = st.file_uploader(
        "Choose files (PDF, DOC, DOCX, TXT)",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True
    )

    dest = st.text_input(
        "Destination folder",
        value="output_markdown"
    )

    # prepare session state for downloads
    if "downloads" not in st.session_state:
        st.session_state.downloads = []

    if st.button("Start conversion"):
        if not uploaded:
            st.error("No files selected.")
            return

        # reset downloads list
        st.session_state.downloads = []

        out_folder = Path(dest)
        out_folder.mkdir(parents=True, exist_ok=True)

        progress = st.progress(0)
        status = st.empty()

        total = len(uploaded)

        for idx, up in enumerate(uploaded, start=1):
            name = up.name
            status.text(f"Converting {name} ({idx}/{total})")
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp:
                tmp.write(up.getvalue())
                tmp_path = tmp.name

            try:
                md = convert_to_markdown(tmp_path)
                out_file = out_folder / f"{Path(name).stem}.md"
                out_file.write_text(md, encoding="utf-8", errors="replace")

                # store for download
                st.session_state.downloads.append((out_file.name, md))

            except Exception as e:
                st.warning(f"Failed: {name}: {e}")

            progress.progress(idx / total)

        status.text("Conversion done.")
        st.success(f"Saved markdown files to {out_folder.resolve()}")

    # show download buttons after conversion
    if st.session_state.downloads:
        st.markdown("### Download Converted Files")
        for name, md in st.session_state.downloads:
            st.download_button(
                label=f"Download {name}",
                data=md,
                file_name=name,
                mime="text/markdown",
                key=f"dl_{name}"
            )


if __name__ == "__main__":
    main()
