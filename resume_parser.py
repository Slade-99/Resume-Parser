import os
import json
import re
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime

from dotenv import load_dotenv
from mistralai import Mistral
import PyPDF2


FIXED_PROMPT_TEMPLATE = """
You are an expert resume parser. Analyze the given resume text carefully and extract all relevant information into the following strictly defined JSON schema.

Always follow these rules:
- Return ONLY the JSON object, without any explanation or markdown formatting.
- The structure and key names must remain exactly as shown below — no additions, deletions, or renaming.
- If any field is not present or unclear, set its value to null.
- If a field expects multiple values, always return an array (even if only one value exists).
- Any extra or miscellaneous information that does not clearly belong to any defined field must be placed inside the "extras" field as a list of text snippets.
- Do NOT include markdown formatting, code fences, or commentary.

### FIXED JSON STRUCTURE:

{
  "name": null,
  "email": null,
  "phone": null,
  "location": null,
  "current_position": null,
  "current_company": null,
  "previous_positions": [
    {
      "company": null,
      "title": null,
      "duration": null
    }
  ],
  "education": [
    {
      "degree": null,
      "institution": null,
      "year": null
    }
  ],
  "skills": [],
  "certifications": [],
  "languages": [],
  "experience_years": null,
  "summary": null,
  "linkedin": null,
  "github": null,
  "portfolio": null,
  "achievements": [],
  "extras": []
}

Now, extract and populate the JSON fields as per the schema above based on the following resume text:

Resume Text:
<<RESUME_TEXT>>

Return ONLY the JSON object exactly in the format specified above.
""".strip()


ALLOWED_TOP_LEVEL_KEYS = [
    "name",
    "email",
    "phone",
    "location",
    "current_position",
    "current_company",
    "previous_positions",
    "education",
    "skills",
    "certifications",
    "languages",
    "experience_years",
    "summary",
    "linkedin",
    "github",
    "portfolio",
    "achievements",
    "extras",
]


class UnifiedResumeParser:
    """
    Unified parser:
      - PDF mode: extract text with PyPDF2
      - Image mode: OCR with Mistral OCR for one or multiple images
      - Shared: structured extraction with Mistral chat using the fixed JSON schema prompt
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ocr_model: str = "mistral-ocr-latest",
        text_model: str = "mistral-large-latest",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        max_retries: int = 3,
    ) -> None:
        load_dotenv()
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY in environment or constructor")

        self.client = Mistral(api_key=api_key)
        self.ocr_model = ocr_model
        self.text_model = text_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # -----------------------------
    # Text extraction (PDF: PyPDF2)
    # -----------------------------
    def _extract_pdf_text_pypdf(self, pdf_path: str) -> str:
        text = []
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i in range(len(reader.pages)):
                    page = reader.pages[i]
                    # PyPDF2 extract_text may return None; guard against it
                    page_text = page.extract_text() or ""
                    text.append(page_text)
        except Exception as e:
            print(f"[WARN] PDF extraction error: {e}")
        return "\n".join(text).strip()

    # --------------------------------------
    # Text extraction (Images: Mistral OCR)
    # --------------------------------------
    def _extract_image_text_mistral(self, image_path: str) -> str:
        try:
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                # Fallback
                mime_type = "application/octet-stream"
            with open(image_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            data_url = f"data:{mime_type};base64,{b64}"

            ocr_response = self.client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "image_url",
                    "image_url": data_url,
                },
            )
            parts = []
            for page in getattr(ocr_response, "pages", []) or []:
                md = getattr(page, "markdown", None)
                if md:
                    parts.append(md)
            return "\n".join(parts).strip()
        except Exception as e:
            print(f"[WARN] Image OCR error for '{image_path}': {e}")
            return ""

    # ------------------------------------------------
    # Structured extraction (Mistral chat completion)
    # ------------------------------------------------
    def _structured_extract(self, full_text: str) -> Dict[str, Any]:
        prompt = FIXED_PROMPT_TEMPLATE.replace("<<RESUME_TEXT>>", full_text)
        messages = [{"role": "user", "content": prompt}]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.complete(
                    model=self.text_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content

                # Extract JSON object via regex
                m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if not m:
                    # No JSON returned; return empty dict to trigger cleaning fallback
                    return {}
                json_str = m.group(0)

                parsed = json.loads(json_str)
                return parsed
            except Exception as e:
                last_err = e
        if last_err:
            print(f"[WARN] Structured extraction failed after retries: {last_err}")
        return {}

    # ---------------------------------
    # Cleaning and schema normalization
    # ---------------------------------
    def _normalize_and_validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Start with all keys present; null or empty list defaults as per schema
        normalized: Dict[str, Any] = {
            "name": data.get("name"),
            "email": data.get("email"),
            "phone": data.get("phone"),
            "location": data.get("location"),
            "current_position": data.get("current_position"),
            "current_company": data.get("current_company"),
            "previous_positions": data.get("previous_positions", []),
            "education": data.get("education", []),
            "skills": data.get("skills", []),
            "certifications": data.get("certifications", []),
            "languages": data.get("languages", []),
            "experience_years": data.get("experience_years"),
            "summary": data.get("summary"),
            "linkedin": data.get("linkedin"),
            "github": data.get("github"),
            "portfolio": data.get("portfolio"),
            "achievements": data.get("achievements", []),
            "extras": data.get("extras", []),
        }

        # Email cleanup
        if normalized["email"]:
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            m = re.search(email_pattern, str(normalized["email"]))
            normalized["email"] = m.group(0) if m else normalized["email"]

        # Phone cleanup (keep a plausible international sequence)
        if normalized["phone"]:
            phone_pattern = r"[+]?[\d][\d\s().-]{6,}"  # permissive pattern
            m = re.search(phone_pattern, str(normalized["phone"]))
            if m:
                candidate = m.group(0)
                # Trim spaces
                candidate = re.sub(r"\s+", " ", candidate).strip()
                normalized["phone"] = candidate

        # Ensure list fields are lists
        list_fields = [
            "previous_positions",
            "education",
            "skills",
            "certifications",
            "languages",
            "achievements",
            "extras",
        ]
        for field in list_fields:
            v = normalized[field]
            if v is None:
                normalized[field] = []
            elif not isinstance(v, list):
                normalized[field] = [v]

        # Enforce item structure for arrays of objects
        def ensure_obj_keys(item: Any, keys: List[str]) -> Dict[str, Any]:
            if not isinstance(item, dict):
                return {k: None for k in keys}
            return {k: item.get(k) for k in keys}

        normalized["previous_positions"] = [
            ensure_obj_keys(it, ["company", "title", "duration"])
            for it in normalized["previous_positions"]
        ] or [{"company": None, "title": None, "duration": None}]

        normalized["education"] = [
            ensure_obj_keys(it, ["degree", "institution", "year"])
            for it in normalized["education"]
        ] or [{"degree": None, "institution": None, "year": None}]

        # Drop any unexpected keys if they appeared
        normalized = {k: normalized.get(k) for k in ALLOWED_TOP_LEVEL_KEYS}

        return normalized

    # ---------------
    # Public API
    # ---------------
    def parse_resume(
        self,
        mode: str,
        inputs: Union[str, Sequence[str]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        mode:
          - 'pdf': inputs is a single path (str) to one PDF
          - 'image': inputs is a str path or a list/sequence of image paths
        Returns the normalized JSON dict; optionally writes to output_path.
        """
        mode_l = (mode or "").strip().lower()
        if mode_l not in {"pdf", "image"}:
            raise ValueError("mode must be 'pdf' or 'image'")

        source_files: List[str] = []
        combined_text = ""

        if mode_l == "pdf":
            if not isinstance(inputs, str):
                raise TypeError("For mode='pdf', inputs must be a single file path (str)")
            pdf_path = inputs
            source_files = [pdf_path]
            combined_text = self._extract_pdf_text_pypdf(pdf_path)
        else:
            # image mode
            if isinstance(inputs, str):
                image_paths = [inputs]
            else:
                image_paths = list(inputs)
            source_files = image_paths[:]
            texts = []
            for p in image_paths:
                t = self._extract_image_text_mistral(p)
                if t:
                    texts.append(t)
            # Combine all image texts together for a single final JSON
            combined_text = "\n\n--- Page Break ---\n\n".join(texts)

        if not combined_text.strip():
            print("[WARN] No text extracted; returning empty result")
            result: Dict[str, Any] = {k: (None if k not in {"previous_positions", "education", "skills", "certifications", "languages", "achievements", "extras"} else []) for k in ALLOWED_TOP_LEVEL_KEYS}
            return result

        raw = self._structured_extract(combined_text)
        clean = self._normalize_and_validate(raw)

        # Attach metadata (not part of the fixed schema; keep separate container)
        metadata = {
            "input_mode": mode_l,
            "source_files": source_files,
            "extraction_timestamp": datetime.now().isoformat(timespec="seconds"),
            "ocr_model": self.ocr_model if mode_l == "image" else None,
            "text_model": self.text_model,
        }

        final_out = {
            **clean,
            "_metadata": metadata,  # kept separate so the fixed schema stays identical
        }

        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_out, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] Failed to write output file: {e}")

        return final_out



if __name__ == "__main__":
    parser = UnifiedResumeParser()
#     # PDF example:
    result = parser.parse_resume("pdf", "Input_Data/pdf/client_1/1720537952266.pdf - Habib Ulla.pdf", output_path="resume_pdf.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
#     # Image(s) example:
    result = parser.parse_resume("image", ["Input_Data/image/client_1/page_1.png", "Input_Data/image/client_1/page_2.png"], output_path="resume_img.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
