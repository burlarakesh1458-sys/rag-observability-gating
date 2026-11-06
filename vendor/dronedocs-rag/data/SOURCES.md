# Corpus sources & licenses

All documents were downloaded on 2026-11-06 from public URLs. Raw files are committed under
`data/raw/` so the golden set is reproducible; the derived index (`data/index/`) is gitignored.

| File | Source | License / status |
|---|---|---|
| `pdf/drone_rules_2021.pdf` | Drone Rules, 2021, G.S.R. 589(E), Gazette of India, 25 Aug 2021 (copy hosted at https://www.livelaw.in/pdf_upload/229221-399307.pdf; official: https://egazette.gov.in/WriteReadData/2021/229221.pdf) | Government of India official gazette (public document) |
| `pdf/drone_amendment_rules_2022.pdf` | Drone (Amendment) Rules, 2022, G.S.R. 108(E), 11 Feb 2022 — https://digitalsky.aai.aero/pdf/Drone_Rules_2022.pdf | Government of India official gazette |
| `pdf/drone_amendment_rules_2023.pdf` | Drone (Amendment) Rules, 2023, G.S.R. 715(E), 27 Sep 2023 — http://www.civilaviation.gov.in/sites/default/files/2023-10/Drone%20%28Amendment%29%20Rules%2C%202023.pdf | Government of India official gazette (scanned; OCR text layer) |
| `pdf/pib_drone_rules_backgrounder.pdf` | PIB backgrounder "The Drone Rules, 2021", 28 Jan 2022 — https://static.pib.gov.in/WriteReadData/specificdocs/documents/2022/jan/doc202212810701.pdf | Press Information Bureau, Government of India |
| `html/ardupilot_copter_*.html` | ArduPilot Copter documentation, https://ardupilot.org/copter/docs/<page>.html | CC BY-SA 3.0 (ArduPilot wiki) |
| `markdown/px4_*.md` | PX4 User Guide sources, https://github.com/PX4/PX4-Autopilot/tree/main/docs/en | CC BY 4.0 |

The Gazette PDFs are bilingual (Hindi first, English second); the loader keeps Latin-script lines only
(`chunking.drop_non_latin_lines`). Citations point to the PDF page number of the English text.
