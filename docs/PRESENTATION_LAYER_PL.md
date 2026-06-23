# Warstwa prezentacji virtRTG

Ten dokument opisuje aktualna warstwe prezentacji wtyczki `virtRTG` po
wydzieleniu jej z obiektu `VirtualXRay` do osobnego obiektu `DetectorImage`.
Najwazniejsza zmiana architektoniczna jest taka, ze:

- `VirtualXRay` odpowiada za geometrie, fizyke, uruchomienie projekcji i cache,
- `DetectorImage` odpowiada za mapowanie tonalne, overlaye, krzywe transferu,
  podglad 2D i eksport widoku.

Dokument odnosi sie przede wszystkim do:

- `virtualXRay.py`
- `detectorImage.py`
- `gui/propVirtualXRay.py`
- `gui/propDetectorImage.py`
- `gui/transferCurveDialog.py`
- `xray/xrayPresentation.py`
- `sceneFormat.py`

## 1. Miejsce prezentacji w pipeline

Aktualny pipeline ma piec logicznych etapow:

1. `Geometry` ustawia uklad zrodlo-detektor, tryb `cone` albo `parallel`,
   probkowanie i ewentualne `depth window`.
2. Zrodla (`Volumetric`, `Mesh`) dostarczaja skalar, geometrie i lokalne
   nadpisania materialowe.
3. `Physics` zamienia skalar na `mu`, wykonuje calkowanie po promieniu i
   opcjonalnie przelicza wynik na intensywnosc.
4. `VirtualXRay` zapisuje wynik do cache:
   - `last_line_integral_projection`
   - `last_raw_projection`
   - `last_source_projections`
   - `last_projected_annotations`
5. `DetectorImage` bierze jeden aktywny obraz z pakietu projekcji i dopiero
   wtedy stosuje warstwe prezentacji.

Praktyczna konsekwencja:

- zmiana prezentacji nie uruchamia ray marchingu od nowa,
- `Update view` dziala na cache,
- prezentacja i overlaye nie modyfikuja `last_raw_projection`.

## 2. Co jest wejsciem do prezentacji

Wejsciem dla `DetectorImage` jest 2D tablica `float32`. Moze ona reprezentowac:

- wynik kompozytowy `raw`,
- wynik kompozytowy `line_integral`,
- warstwe per-zrodlo `raw`,
- warstwe per-zrodlo `line_integral`.

Pakiet projekcji jest trzymany w:

- `package_images`
- `package_layers`
- `package_metadata`
- `active_layer_key`

To oznacza, ze uzytkownik moze ogladac nie tylko finalny obraz detektora, ale
tez warstwy poszczegolnych zrodel i obraz calki oslabienia.

## 3. Gdzie sa teraz kontrolki prezentacji

Kontrolki prezentacji nie sa juz zakladka `Presentation` w panelu
`VirtualXRay`.

Aktualny workflow GUI jest nastepujacy:

1. W panelu `VirtualXRay` uzytkownik ustawia `Geometry` i `Physics`.
2. W zakladce `Run` uruchamia `Run simulation`.
3. Plugin tworzy lub aktualizuje obiekt `DetectorImage`.
4. Wlasciwe strojenie obrazu odbywa sie w panelu `DetectorImage`.

W panelu `VirtualXRay` pozostala tylko informacja:

- "Presentation controls are handled by the generated DetectorImage object."

To jest zgodne z aktualna architektura kodu.

## 4. Tryby prezentacji

Implementacyjnie nadal istnieja trzy tryby:

- `raw`
- `digital`
- `film`

### 4.1. `raw`

W `DetectorImage` tryb `raw` nie oznacza "pokaz surowe liczby bez zadnej
obrobki". W praktyce:

- aktywne jest okno `window_center` / `window_width`,
- wynik jest normalizowany do `[0, 1]`,
- opcjonalnie mozna wykonac `invert`,
- nie sa stosowane `gamma`, `contrast`, `input_transform` ani `CLAHE`.

Ten tryb sluzy glownie do inspekcji danych detektora.

Kolejnosc wykonywania w trybie `raw` jest nastepujaca:

1. Pobranie `raw_array` z aktywnej warstwy.
2. Wyznaczenie aktywnego okna:
   - z `window_center` i `window_width`, albo
   - z automatycznego zakresu percentylowego, jesli szerokosc okna nie jest
     dodatnia.
3. Normalizacja do `[0, 1]` w ramach tego okna.
4. Opcjonalne `invert`.
5. Zwracany jest obraz "presented".

Po tym wspolnym etapie `DetectorImage` wykonuje jeszcze:

6. Opcjonalna krzywa transferu `transfer_points_pct`.
7. Opcjonalne `display_only_window_range`, czyli wyzerowanie pikseli poza
   oknem.
8. Obciecie wyniku do `[0, 1]`.
9. Przy renderze `QImage` dorysowanie overlayow i etykiet.

### 4.2. `digital`

To podstawowy tryb roboczy. Stosuje:

1. Wybor zakresu:
   - jawne okno `window_center` / `window_width`, albo
   - automatyczny zakres z percentyli
     `robust_low_percentile` i `robust_percentile`.
2. Opcjonalny `input_transform`:
   - `linear`
   - `log1p`
3. Odpowiednia normalizacje do `[0, 1]`.
4. Opcjonalne `invert`.
5. `contrast`.
6. `gamma`.
7. Opcjonalne lokalne wzmocnienie:
   - `off`
   - `clahe`

Istotna roznica wzgledem starszej dokumentacji:

- automatyczne okno jest teraz dwustronne,
- dolna granica nie jest juz na stale `min(image)`,
- panel ma osobne sterowanie `robust_low_percentile`.

Kolejnosc wykonywania w trybie `digital` jest nastepujaca:

1. Pobranie `raw_array` z aktywnej warstwy.
2. `input_transform`:
   - `linear`, albo
   - `log1p`.
3. Wyznaczenie zakresu:
   - z `window_center` i `window_width`, jesli okno jest jawnie ustawione,
   - w przeciwnym razie z percentyli
     `robust_low_percentile` i `robust_percentile`.
4. Normalizacja do `[0, 1]`.
5. Opcjonalne lokalne wzmocnienie:
   - brak,
   - `clahe`.
6. Opcjonalne `invert`.
7. `contrast`.
8. `gamma`.
9. Zwracany jest obraz "presented".

Po tym wspolnym etapie `DetectorImage` wykonuje jeszcze:

10. Opcjonalna krzywa transferu `transfer_points_pct`.
11. Opcjonalne `display_only_window_range`.
12. Obciecie wyniku do `[0, 1]`.
13. Przy renderze `QImage` dorysowanie overlayow i etykiet.

### 4.3. `film`

`film` nadal jest modelem prezentacyjnym, a nie fizycznym modelem kliszy.
Korzysta z tego samego zestawu rozszerzen co `digital`:

- `input_transform`
- `contrast`
- `gamma`
- `invert`
- opcjonalne `clahe`

W odroznieniu od `digital`, `film` opiera sie na normalizacji percentylowej,
bez recznego okna jako glownego mechanizmu pracy.

Kolejnosc wykonywania w trybie `film` jest nastepujaca:

1. Pobranie `raw_array` z aktywnej warstwy.
2. `input_transform`:
   - `linear`, albo
   - `log1p`.
3. Wyznaczenie zakresu z percentyli:
   - `robust_low_percentile`
   - `robust_percentile`
   albo z `fixed_range`, jesli taki zakres zostal podany przez model.
4. Normalizacja do `[0, 1]`.
5. Opcjonalne lokalne wzmocnienie:
   - brak,
   - `clahe`.
6. Opcjonalne `invert`.
7. `contrast`.
8. `gamma`.
9. Zwracany jest obraz "presented".

Po tym wspolnym etapie `DetectorImage` wykonuje jeszcze:

10. Opcjonalna krzywa transferu `transfer_points_pct`.
11. Opcjonalne `display_only_window_range`.
12. Obciecie wyniku do `[0, 1]`.
13. Przy renderze `QImage` dorysowanie overlayow i etykiet.

## 5. Kolejnosc globalna w `DetectorImage`

Niezaleznie od trybu warto patrzec na warstwe prezentacji jako na dwa poziomy:

1. Bazowy model prezentacji:
   - `raw`
   - `digital`
   - `film`
2. Etapy wspolne wykonywane po nim:
   - krzywa transferu,
   - opcjonalna maska `display_only_window_range`,
   - konwersja do widoku 2D,
   - overlaye i etykiety.

Skrotowo:

```text
raw_array
-> model raw/digital/film
-> transfer curve
-> optional window mask
-> clip [0,1]
-> QImage
-> overlays
```

## 6. Dodatkowe etapy po prezentacji bazowej

Po uzyskaniu obrazu "presented" `DetectorImage` wykonuje jeszcze dwa kroki,
ktorych starsza dokumentacja nie opisywala:

### 5.1. Krzywa transferu

Obiekt przechowuje:

- `transfer_points_pct`

Domyslnie:

```text
[(0.0, 0.0), (100.0, 100.0)]
```

Jest to dodatkowa, odcinkowo-liniowa krzywa w przestrzeni juz
znormalizowanej. Po prezentacji bazowej wykonywane jest `interp(...)`
na punktach procentowych. To pozwala:

- recznie rozciagnac wybrane zakresy tonalne,
- przyciac zakresy bez zmiany fizyki,
- dopracowac wyglad do publikacji lub prezentacji.

### 5.2. `display_only_window_range`

Jesli ta opcja jest aktywna, piksele spoza aktywnego okna sa zerowane po
zastosowaniu mapowania tonalnego. To nie zmienia surowych danych, tylko
koncowy obraz wyswietlany.

## 7. Overlaye i adnotacje

Overlaye sa kompozytowane przez `DetectorImage`, a nie mieszane z surowym
buforem projekcji.

Aktualnie wspierane sa co najmniej:

- krzyzyki (`XRayOverlayCross`)
- polilinie (`XRayOverlayPolyline`)

Uzytkownik ma do dyspozycji:

- `Show projected annotations`
- `Show annotation labels`
- `Cross size [px]`

oraz osobne okno:

- `Edit overlays`

To okno pozwala edytowac punktowe overlaye w przestrzeni detektora:

- etykiete,
- wspolrzedne `U/V`,
- kolor.

## 8. Nowe okna i edytory

### 7.1. `Open detector`

W panelu `VirtualXRay` przycisk `Open detector` otwiera osobne okno 2D dla
aktywnego obiektu `DetectorImage`.

To okno sluzy do:

- wygodnego ogladania obrazu detektora,
- szybkiego porownywania warstw,
- pracy na obrazie poza widokiem 3D workspace.

### 7.2. `TransferCurveDialog`

Przycisk `Edit curve` w panelu `DetectorImage` otwiera okno edycji krzywej
transferu. Dialog zawiera:

- liste punktow krzywej,
- podglad krzywej,
- histogram wejscia do krzywej,
- przeciaganie punktow myszka,
- import i eksport punktow do pliku tekstowego.

W przypadku `DetectorImage` histogram budowany jest z obrazu po prezentacji
bazowej, ale przed dodatkowa krzywa transferu.

### 7.3. Edycja krzywej odpowiedzi materialowej

To trzeba odroznic od krzywej transferu obrazu.

W zakladce `Physics` panelu `VirtualXRay` przycisk `Edit custom curve` otwiera
edytor krzywej `HU -> mu` dla `physics_material_response_curve_points`.

Sa to dwa rozne poziomy:

- krzywa materialowa dziala przed projekcja,
- krzywa transferu `DetectorImage` dziala po projekcji.

## 9. Parametry panelu `DetectorImage`

Najwazniejsze pola to:

- `Layer`
- `Mode`
- `Window center`
- `Window width`
- `Gamma`
- `Contrast`
- `Input transform`
- `Local enhancement`
- `CLAHE clip`
- `CLAHE tile`
- `Robust [%]` z osobnym `Low` i `High`
- `Only window range`
- `Invert`
- `Show projected annotations`
- `Show annotation labels`
- `Cross size [px]`

Akcje:

- `Auto window`
- `Full range`
- `Show 2D`
- `Edit curve`
- `Edit overlays`
- `Save PNG`
- `Import array`
- `Export array`

## 10. Presety prezentacji

Presety sa przechowywane jako `detector_image_defaults` w obiekcie
`VirtualXRay`, ale stosowane przez generowany `DetectorImage`.

Lista presetow:

- `default`
- `balanced`
- `bone_soft`
- `bone_contrast`
- `film_soft`

Aktualny zestaw zawiera juz nie tylko `mode`, `gamma`, `contrast` i `window`,
ale takze:

- `input_transform`
- `local_enhancement`
- `clahe_clip_limit`
- `clahe_tile_grid_size`
- `robust_low_percentile`

W szczegolnosci:

- `bone_soft` uzywa `input_transform = log1p`,
- `bone_contrast` uzywa `input_transform = log1p` oraz `local_enhancement = clahe`.

To jest wazna roznica wzgledem starszego opisu presetow.

## 11. Cache, eksport i import

### 10.1. Eksport cache projekcji

W `VirtualXRay` mozna eksportowac cache:

- `raw`
- `line_integral`

do formatow:

- `.npz`
- `.npy`
- `.txt`
- `.csv`
- `.tsv`

Jesli zapis odbywa sie do `.npz`, tworzony jest bogatszy pakiet
`virtRTG-detector-package`, ktory moze zawierac:

- obraz kompozytowy,
- obrazy per-zrodlo,
- warstwy `raw` i `line_integral`,
- ustawienia prezentacji,
- krzywa transferu,
- overlaye,
- fragment metadanych symulacji.

### 10.2. `DetectorImage` jako warstwa robocza

`DetectorImage` potrafi:

- importowac pakiet `.npz`,
- importowac pojedyncze tablice `.npy` lub tekstowe,
- zachowac ustawienia prezentacji po round-trip,
- przelaczac aktywna warstwe `Layer`.

To czyni z niego samodzielny obiekt do analizy obrazu detektora, niezaleznie
od tego, czy projekcja zostala policzona lokalnie, czy wczytana z pliku.

## 12. Zapis do ATMDL

Po zmianach warto podkreslic rozdzial odpowiedzialnosci przy zapisie `ATMDL`:

- `VirtualXRay` jest eksportowany do bloku `virtualXRay { ... }`,
- zapisywane sa geometria, ustawienia zrodel, fizyka i
  `detectorImageDefaults`,
- zapisywane sa tez sekcje tekstowe przyjazne do recznej edycji,
- dodatkowo istnieje payload `virtRTGConfig64` do pelniejszego round-tripu.

Obecnie `DetectorImage` nie jest zapisywany jako osobny obiekt do `ATMDL`.
Jego stan roboczy sluzy glownie do:

- biezacej inspekcji,
- eksportu `PNG`,
- eksportu pakietow projekcji `.npz`.

W praktyce:

- `ATMDL` sluzy do zapisu sceny i ustawien `VirtualXRay`,
- `.vxrscene.xml` sluzy do pelniejszego zapisu poddrzewa pluginu,
- `.npz` sluzy do wymiany obrazow detektora i pakietow projekcji.

## 13. Ograniczenia interpretacyjne

- Warstwa prezentacji nie jest modelem klinicznego detektora.
- `film` nie symuluje fizyki kliszy.
- `CLAHE`, `log1p` i krzywa transferu sa narzedziami wizualizacji, nie fizyki.
- Overlaye i podpisy sa warstwa prezentacyjna.
- Zmiana `DetectorImage` nie powinna byc interpretowana jako zmiana wyniku
  obliczen projekcyjnych.
- Do porownan ilosciowych nalezy uzywac danych `raw` albo
  `line_integral`, a nie obrazu po prezentacji.
