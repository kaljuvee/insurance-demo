# IF Kindlustuse AI Demo — Kasutaja Juhend

Versioon: 1.0

## Ülevaade
Kindlustuse Kahjunõuete Analüsaator aitab analüüsida nõudeid ja arveid poliisipakkumistega ning hinnata pilte sisseehitatud pettuse tuvastamisega.

- Nõuete dokumentide väljavõtmine ja võrdlemine
- Arvetoetuse tugi (PDF ja pildiarved)
- Pettuse tuvastamine dokumentide ja piltide jaoks
- Pildi kahjustuste analüüs vale positiivsete/negatiivsete skooridega
- Eksportitavad JSON-aruanded

## Nõuded
- Python 3.10+
- OpenAI API võtme(kohustuslik)
- Anthropic API võtme(valikuline)
- Soovitatav: `pandoc` Markdown → PDF eksportimiseks

## Seadistamine
1) Paigaldage ja käivitage
```bash
cd /home/julian/dev/hobby/insurance-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```
2) Looge `.env` fail
```bash
OPENAI_API_KEY=teie_openai_api_võti
ANTHROPIC_API_KEY=teie_anthropic_api_võti   # valikuline
```

## Andmete Kaust
Asetage dokumendid ja pildid kausta `data/` (alamkaustade tugi):
- PDF-d: nõuded, arved, poliisipakkumised
- Pildid: `.jpg`, `.jpeg`, `.png`

## Lehed
### Nõuete Kontroll
- Ekstraktige andmed Nõudest, Arvest, Poliisist
- Võrdlege kõrvuti erinevuste ja soovitustega
- Toetatakse pildiarveid (analüüsige ehtsust läbi Pettuse Tuvastamise)

Töövoog:
1. Lisage API võti (külgriba või `.env` kaudu).
2. Kasutage demo-faile või laadige üles PDF-/pilte.
3. Ekstrakteerige Nõue/Arve/Poliis.
4. Valikuline: lisage salvestatud pildi analüüsid.
5. Võrrelge, et näha erisusi ja eksportige aruanne.

### Pettuse Tuvastus (Nõuete Kontrolli sees)
- Valige fail kaustast `data/` (PDF/pilt) külgriba kaudu
- Tuvastage muudetud/petturlik sisu ja manipuleerimine
- Väljund: riskiskoor, ehtsus, näitajad, soovitused
- Eksportige JSON aruanne

### Pildi Tuvastus
- Analüüsi režiimid: Üksikasjalik, Kahjustuste Hindamine, Pettuse Tuvastus, Vale Positiivne/Negatiivne, OCR
- Kasutage kaasaskantavaid pilte kaustast `data/` või laadige üles oma
- Salvestage analüüsid Nõuete Kontrollis kasutamiseks
- Visualiseerige tõsidus/täpsus ja eksportige tulemused

### Dokumentatsioon
- Laadige alla see juhend (Markdown/PDF)
- Sammud PDFi kohalikuks genereerimiseks

## API Võtmed
- OpenAI vajalik; Anthropic valikuline
- Laadige `.env` failist ja/või üle kirjutage külgribadel

## Eksport
- Nõuete võrdlemine: JSON
- Pildi analüüs: JSON või tekst
- Pettuse tuvastamine: JSON
- Kasutaja juhend: `docs/if-insurance-ai-demo-user-guide.(md|pdf)`

## PDFi Regeneratsioon
Kui `pandoc` on paigaldatud:
```bash
cd /home/julian/dev/hobby/insurance-demo
source .venv/bin/activate
pandoc docs/if-insurance-ai-demo-user-guide.md -o docs/if-insurance-ai-demo-user-guide.pdf
```
Kui puudub (Debian/Ubuntu):
```bash
sudo apt-get update && sudo apt-get install -y pandoc
```

## Tõrkeotsing
- API võti on vajalik: lisage kaudu `.env` või külgriba
- PDF ei genereerita: paigaldage `pandoc`
- Suurtes failides aeglane: proovige väiksemaid sisendeid
- Anthropic mudel ilma võtmeta: vahetage OpenAI vastu või lisage võti

## Privaatsus ja Turvalisus
- Vältige tundlike andmete üleslaadimist ilma nõuetekohaste lepingute/nõusolekuta
- Kaaluge isiklike andmete redigeerimist

© IF Kindlustuse AI Demo — Ainult demonstratsiooniks.