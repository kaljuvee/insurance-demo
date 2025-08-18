# Kindlustuskaebuste Analüsaator

Streamlit rakendus, mis aitab kindlustusprofessionaalidel ja poliiside omanikel analüüsida ja võrrelda kindlustuskaebusi poliisikokkulepetega kasutades tehisintellekti.

## Funktsioonid

- **Kaebuste Kontroll**: Laadi üles arve/kaebusdokumente ja võrrelge neid poliisikokkulepetega
- **Automaatne Ekstraktsioon**: Kasutab OpenAI keelemudeleid, et välja tõmmata vajalikku teavet PDF-dokumentidest
- **Kõrvuti Võrdlus**: Lihtsalt tuvasta erinevused kaebuste ja poliisi katte vahel
- **Esile Toodud Erinevused**: Kiiresti tuvastage potentsiaalsed probleemid või murekohad

## Paigaldamine

1. Klonige see repositoorium:
   ```
   git clone https://github.com/kaljuvee/insurance-demo.git
   cd insurance-demo
   ```

2. Installige vajalikud sõltuvused:
   ```
   pip install -r requirements.txt
   ```

3. Käivitage Streamlit rakendus:
   ```
   streamlit run Home.py
   ```

## Kasutamine

1. Ava rakendus ja navigeeri antud URL-ile (tavaliselt http://localhost:8501)
2. Sisestage oma OpenAI API võti külgribale
3. Navigeerige külgribal "Kaebuste Kontroll" lehele
4. Laadi üles oma arve/kaebusdokument
5. Laadi üles vastav kindlustuspoliisi leping
6. Klõpsake "Ekstraheeri Kaebuse Andmed" ja "Ekstraheeri Poliiandi Andmed" dokumentide töötlemiseks
7. Klõpsake "Võrdle Dokumente", et genereerida üksikasjalik võrdlus
8. Vaadake kõrvuti võrdlust ja esile toodud erinevusi

## Näidisdokumendid

`data` kataloog sisaldab katsetamiseks näidiskindlustusdokumente:
- `invoice-kristjan-tamm-001.pdf` - Näidisarve
- `invoice-kristjan-tamm-002.pdf` - Veel üks näidisarve
- `insurance-claim-kristjan-tamm-001.pdf` - Näidiskindlustuskaebus
- `insurance-claim-kristjan-tamm-002.pdf` - Veel üks näidiskindlustuskaebus
- `insurance-contract-kristjan-tamm.pdf` - Näidiskindlustuspoliisi leping

## Nõuded

- Python 3.8+
- OpenAI API võti
- Sõltuvused, mis on loetletud requirements.txt failis

## Kuidas See Töötab

1. **PDF Teksti Ekstraktsioon**: Rakendus ekstraktsioonib tekst PDF-dokumentidest, kasutades PyPDF2
2. **Tehisintellekti Toega Analüüs**: OpenAI keelemudelid ekstraktsioonivad struktureeritud andmeid dokumentidest
3. **Intelligentne Võrdlus**: Rakendus võrdleb ekstraktsioonitud andmeid, et tuvastada erinevused
4. **Visuaalsed Tulemused**: Tulemused esitatakse arusaadaval kujul esile toodud erinevustega

## Litsents

See projekt on litsentseeritud repositooriumis sisalduvate litsentsitingimuste alusel.