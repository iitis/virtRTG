# Warstwa prezentacji virtRTG

Ten dokument opisuje warstwe prezentacji wtyczki `virtRTG` z perspektywy
uzytkownika i implementacji. Skupia sie na tym, jak z surowego wyniku
projekcji powstaje obraz wyswietlany w `pyDpVision`, jakie parametry sa
dostepne w GUI, jakie wzory sa stosowane oraz czym roznia sie gotowe presety.

Dokument odnosi sie przede wszystkim do:

- `gui/propVirtualXRay.py`
- `virtualXRay.py`
- `xray/xrayPresentation.py`
- `xray/xrayProjection.py`

## 1. Miejsce warstwy prezentacji w pipeline

Pipeline w `virtRTG` ma cztery logiczne etapy:

1. Geometria ustala uklad zrodlo-detektor i sposob probkowania promieni.
2. Zrodla (`Volumetric`, `Mesh`) dostarczaja wartosci skalarne lub oslabienia.
3. Model fizyczny zamienia skalar na wspolczynnik oslabienia i integruje go
   wzdluz promienia.
4. Warstwa prezentacji zamienia wynik numeryczny na obraz do ogladania.

Najwazniejsza konsekwencja praktyczna:

- presety prezentacji nie zmieniaja geometrii,
- presety prezentacji nie zmieniaja fizyki projekcji,
- presety prezentacji zmieniaja tylko mapowanie tonalne obrazu wynikowego.

To oznacza, ze `default`, `bone_soft` i `bone_contrast` korzystaja z tego
samego surowego obrazu, ale pokazuja go inaczej.

## 2. Co jest wejsciem do prezentacji

Domyslnie `VirtualXRay` pracuje z:

- `physics_output_mode = "integral"`
- `presentation_mode = "digital"`

W takim trybie warstwa prezentacji dostaje jako wejscie sume oslabienia
wzdluz promienia:

```text
L = integral(mu(s) ds)
```

W implementacji dyskretnej jest to przyblizane przez sumowanie po krokach:

```text
L ~= sum(mu_i * step_mm)
```

Im wieksza wartosc `L`, tym "gestsza" radiologicznie sciezka promienia.

Jesli uzytkownik przelaczy:

- `physics_output_mode = "intensity"`

to do warstwy prezentacji trafia juz obraz intensywnosci:

```text
I = exp(-L)
```

czyli klasyczna uproszczona postac prawa Beer-Lamberta.

## 3. Wzory uzywane przed prezentacja

Sama prezentacja nie liczy fizyki, ale jej efekt zalezy od tego, co dostaje na
wejsciu. Dlatego ponizej zebrano wzory z warstwy fizycznej, ktore maja
bezposredni wplyw na obraz.

### 3.1. Domyslne mapowanie HU -> oslabienie

Dla trybu `linear` kod stosuje:

```text
relative_density = max(0, 1 + HU / abs(hounsfield_air))
mu = (mu_air + mu_water * relative_density) * attenuation_scale * energy_scale
```

gdzie:

```text
energy_scale = (reference_energy_kev / source_energy_kev) ^ attenuation_energy_exponent
```

Domyslne wartosci:

- `mu_air = 0.0`
- `mu_water = 0.02`
- `hounsfield_air = -1000.0`
- `attenuation_scale = 1.0`
- `source_energy_kev = 70.0`
- `reference_energy_kev = 70.0`
- `attenuation_energy_exponent = 2.0`

Przy domyslnych energiach `energy_scale = 1`.

### 3.2. Tryby odpowiedzi materialowej

W zakladce `Physics` sa cztery glowne tryby:

- `linear`
- `piecewise_bone`
- `piecewise_soft_tissue`
- `bone_threshold`

#### `piecewise_bone`

To odcinkowo-liniowa krzywa wzmacniajaca zakres kostny. Punkty kontrolne:

```text
HU      mnoznik * base
-1000   mu_air
-300    0.03
0       0.10
150     0.18
400     0.35
800     0.75
1200    1.20
2000    1.85
3000    2.30
4000    2.60
```

gdzie:

```text
base = mu_water * attenuation_scale * energy_scale
```

#### `piecewise_soft_tissue`

To krzywa bardziej plaska w zakresie kosci i relatywnie bardziej przyjazna dla
tkanek miekkich:

```text
HU      mnoznik * base
-1000   mu_air
-300    0.05
0       0.45
80      0.70
200     0.85
500     1.05
1000    1.20
2000    1.35
4000    1.55
```

#### `bone_threshold`

To mieszanie modelu liniowego z modelem `piecewise_bone` przez wage logistyczna:

```text
weight = 1 / (1 + exp(-(HU - threshold) / softness))
mu = linear_mu * (1 - 0.85 * weight) + bone_mu * weight
```

Parametry:

- `threshold` to `physics_bone_threshold_hu`
- `softness` to `physics_bone_threshold_softness`

Jesli prog nie jest ustawiony, backend przyjmuje `350 HU`.

### 3.3. Okno materialowe

Opcjonalne okno materialowe nie zmienia geometrii ani samej prezentacji, ale
zeruje lub oslabia wklad spoza zadanego zakresu HU jeszcze przed integracja.

Granice okna:

```text
vmin = center - width / 2
vmax = center + width / 2
```

Tryby wag:

`hard`

```text
weight = 1 dla HU w [vmin, vmax], w przeciwnym razie 0
```

`linear`

```text
lower = clip((HU - (vmin - softness)) / softness, 0, 1)
upper = clip(((vmax + softness) - HU) / softness, 0, 1)
weight = lower * upper
```

`sigmoid`

```text
lower = 1 / (1 + exp(-(HU - vmin) / softness))
upper = 1 / (1 + exp((HU - vmax) / softness))
weight = lower * upper
```

Na koncu:

```text
mu_out = mu * weight
```

### 3.4. Konwersja integral -> intensity

Jesli `physics_output_mode = "intensity"`:

```text
I = exp(-L)
```

Opcjonalnie moze zostac dodany heurystyczny zysk zalezy od odleglosci zrodla:

```text
G = (d_ref / d) ^ power
I_out = max(intensity_floor, I * G)
```

Jest to aktywne tylko dla:

- `projection_mode = "cone"`
- `physics_source_distance_falloff_mode = "inverse_square"`

Jesli tryb wyjsciowy pozostaje `integral`, a wlaczone jest `inverse_square`,
backend nie mnozy przez `G`, tylko stosuje rownowazne przesuniecie logarytmiczne:

```text
L_out = L - log(G)
```

## 4. Wlasciwa warstwa prezentacji

Implementacyjnie sa trzy tryby:

- `digital`
- `film`
- `raw`

### 4.1. `raw`

`RawPresentationModel` niczego nie mapuje. Zwraca kopie obrazu jako `float32`.

Uwaga praktyczna: w GUI taki obraz i tak musi byc sprowadzony do `uint8`, aby
pokazac go w oknie. Dlatego sam widok roboczy nie jest "matematycznie surowy"
w sensie bufora wyswietlania, ale nie jest wtedy stosowany model
`digital`/`film`.

### 4.2. `digital`

To podstawowy tryb do pracy interaktywnej.

Najpierw wybierany jest zakres normalizacji:

- jesli `window_width > 0`, to:

```text
vmin = window_center - window_width / 2
vmax = window_center + window_width / 2
```

- w przeciwnym razie:

```text
vmin = min(image)
vmax = percentile(image, robust_percentile)
```

Potem wykonywane jest mapowanie:

```text
n = clip((image - vmin) / (vmax - vmin), 0, 1)
```

Jesli wlaczone jest `invert`:

```text
n = 1 - n
```

Nastepnie kontrast:

```text
n = clip(0.5 + (n - 0.5) * contrast, 0, 1)
```

Na koncu gamma:

```text
display = n ^ (1 / gamma)
```

Wynik jest obrazem `float32` w zakresie `[0, 1]`.

### 4.3. `film`

Tryb `film` stosuje dokladnie te sama sekwencje:

- normalizacja,
- opcjonalne odwracanie,
- kontrast,
- gamma.

Roznica wzgledem `digital` jest taka, ze `film` nie ma parametrow
`window_center` i `window_width`. Zawsze opiera sie na:

- minimum obrazu,
- percentylu `robust_percentile`.

W praktyce `film` daje bardziej "gesty" i mniej techniczny wyglad, ale nie
jest fizycznym modelem kliszy.

## 5. Parametry dostepne w zakladce Presentation

### 5.1. Preset

Lista presetow:

- `default`
- `balanced`
- `bone_soft`
- `bone_contrast`
- `film_soft`

Przycisk `Apply` tylko przepisuje zapisane wartosci do aktualnego obiektu
`VirtualXRay`.

### 5.2. Mode

Dostepne opcje:

- `digital`
- `film`
- `raw`

Znaczenie:

- `digital`: normalizacja z opcjonalnym oknem,
- `film`: normalizacja percentylowa bez okna,
- `raw`: bez modelu prezentacji.

### 5.3. Invert

Typ:

- `bool`

Znaczenie:

- odwraca znormalizowany obraz po normalizacji, ale przed kontrastem i gamma.

Praktyka:

- dla obrazu opartego o `integral` ustawienie `invert = False` daje jasniejsze
  struktury o duzym oslabieniu,
- dla obrazu opartego o `intensity` bardziej klasyczny wyglad RTG czesto wymaga
  `invert = True`.

### 5.4. Gamma

Zakres GUI:

- `0.05` do `10.0`

Wzor:

```text
display = n ^ (1 / gamma)
```

Efekt praktyczny:

- `gamma < 1` przyciemnia srednie tony,
- `gamma > 1` rozjasnia srednie tony.

### 5.5. Contrast

Zakres GUI:

- `0.05` do `10.0`

Wzor:

```text
n = clip(0.5 + (n - 0.5) * contrast, 0, 1)
```

Efekt praktyczny:

- `1.0` oznacza brak zmiany,
- wartosci `> 1.0` zwiekszaja separacje tonalna wokol srodka skali,
- wartosci `< 1.0` splaszczaja kontrast.

### 5.6. Robust [%]

Zakres GUI:

- `50.0` do `100.0`

Znaczenie:

- gorny percentyl uzywany jako `vmax`, jesli nie ma aktywnego okna.

Efekt praktyczny:

- nizszy percentyl szybciej obcina skrajnie wysokie wartosci,
- to zwykle wzmacnia widocznosc struktur gestych kosztem pelnej skali
  najjasniejszych pikseli.

### 5.7. Window center

Zakres GUI:

- `-1e6` do `1e6`

Znaczenie:

- srodek zakresu okna dla trybu `digital`.

Uwaga:

- parametr ma znaczenie tylko wtedy, gdy `Window width > 0`.

### 5.8. Window width

Zakres GUI:

- `0.0` do `1e6`

Znaczenie:

- szerokosc okna dla trybu `digital`.

Interpretacja:

- `0.0` oznacza w praktyce brak aktywnego okna,
- wartosc dodatnia przelacza `digital` z trybu percentylowego na sztywne okno.

### 5.9. Overlay projected annotations

Typ:

- `bool`

Znaczenie:

- po wlaczeniu na obrazie rysowane sa zrzutowane adnotacje 2D.

### 5.10. Show labels

Typ:

- `bool`

Znaczenie:

- pokazuje podpisy obok adnotacji.

Warunek:

- opcja ma sens tylko wtedy, gdy wlaczone sa same adnotacje.

### 5.11. Cross size [px]

Zakres GUI:

- `1` do `64`

Znaczenie:

- rozmiar znacznika krzyzowego dla adnotacji punktowych.

## 6. Parametry powiazane z prezentacja, ale z innych zakladek

### 6.1. `Physics -> output_mode`

Opcje:

- `integral`
- `intensity`

To najwazniejszy parametr spoza zakladki `Presentation`, bo zmienia semantyke
obrazu wejsciowego.

### 6.2. `Physics -> distance_falloff`

Opcje:

- `none`
- `inverse_square`

To heurystyka, ktora moze zmienic rozklad jasnosci na detektorze przy projekcji
stozkowej.

### 6.3. `Physics -> Response`

Opcje:

- `linear`
- `piecewise_bone`
- `piecewise_soft_tissue`
- `bone_threshold`

To nie jest prezentacja w scislym sensie, ale mocno zmienia to, co widac po
normalizacji. Jesli celem jest podkreslenie kosci, to zwykle trzeba stroic
zarowno `Response`, jak i preset prezentacji.

### 6.4. `Geometry -> quality`

Opcje:

- `draft`
- `normal`
- `high`
- `custom`

To nie jest warstwa prezentacji, ale wplywa na ostrosc i ilosc aliasingu:

- `draft`: `step_mm = 2.0`, `detector_downsample = 2`
- `normal`: `step_mm = 1.0`, `detector_downsample = 1`
- `high`: `step_mm = 0.5`, `detector_downsample = 1`
- `custom`: zachowuje `step_mm` z obiektu, `detector_downsample = 1`

## 7. Presety prezentacji

### 7.1. Zestawienie liczbowe

| Preset | Mode | Invert | Gamma | Contrast | Robust [%] | Window center | Window width |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `default` | `digital` | `False` | 0.70 | 1.20 | 99.5 | `None` | `None` |
| `balanced` | `digital` | `False` | 0.85 | 1.10 | 99.2 | `None` | `None` |
| `bone_soft` | `digital` | `False` | 1.35 | 1.18 | 98.8 | `None` | `None` |
| `bone_contrast` | `digital` | `False` | 1.55 | 1.40 | 98.4 | `None` | `None` |
| `film_soft` | `film` | `False` | 1.60 | 1.10 | 99.0 | `None` | `None` |

### 7.2. Co robi `default`

`default` jest presetem najszybszym do "czytania" przy standardowym
`output_mode = integral`:

- `gamma = 0.70` przyciemnia srednie tony,
- `contrast = 1.20` podbija separacje tonalna,
- `robust_percentile = 99.5` zostawia dosc szeroki zakres wysokich wartosci.

W praktyce:

- obraz jest bardziej techniczny,
- bardzo gesta tkanka nie jest agresywnie kompresowana,
- kosc jest widoczna, ale nie tak celowo eksponowana jak w presetach `bone_*`.

### 7.3. Czym rozni sie `balanced`

Wzgledem `default`:

- ma wyzsze `gamma` (`0.85` zamiast `0.70`),
- ma nizszy kontrast (`1.10` zamiast `1.20`),
- ma troche nizszy percentyl (`99.2` zamiast `99.5`).

Efekt:

- mniej agresywne przyciemnienie srednich tonow,
- lagodniejszy wyglad,
- bardziej neutralny kompromis miedzy koscia i tkankami miekkimi.

### 7.4. Czym rozni sie `bone_soft`

Wzgledem `default`:

- `gamma` rosnie z `0.70` do `1.35`,
- `contrast` lekko spada z `1.20` do `1.18`,
- `robust_percentile` spada z `99.5` do `98.8`.

Efekt:

- srednie i wyzsze tony sa bardziej rozjasnione,
- najgestsze wartosci sa mocniej kompresowane przez nizszy percentyl,
- kosc staje sie czytelniejsza, ale bez bardzo ostrego "uderzenia".

To preset dobry wtedy, gdy:

- chcesz podkreslic kostne kontury,
- ale nie chcesz przesadnie wyostrzac obrazu.

### 7.5. Czym rozni sie `bone_contrast`

Wzgledem `default`:

- `gamma` rosnie do `1.55`,
- `contrast` rosnie do `1.40`,
- `robust_percentile` spada do `98.4`.

To najbardziej zdecydowany preset kostny w zestawie.

Efekt:

- silne rozjasnienie tonalne w obszarze kosci,
- mocniejsze rozsuniecie tonalne wokol srodka skali,
- szybsza kompresja skrajnych wysokich wartosci.

W praktyce:

- krawedzie i przebieg struktur kostnych sa najbardziej wyeksponowane,
- latwiej stracic subtelne przejscia w mniej gestych tkankach,
- obraz robi sie bardziej "diagnostyczny wizualnie", ale mniej neutralny.

### 7.6. Czym rozni sie `film_soft`

To jedyny preset startowy oparty o `film`, a nie `digital`.

Wzgledem `default`:

- zmienia model prezentacji z `digital` na `film`,
- ma `gamma = 1.60`,
- ma `contrast = 1.10`,
- ma `robust_percentile = 99.0`.

Efekt:

- tonalnosc jest bardziej miekka i mniej "interfejsowa",
- brak okna sprawia, ze preset zawsze pracuje na pelnym zakresie min-percentyl,
- dobrze nadaje sie do porownan wizualnych i obrazow prezentacyjnych.

## 8. Presety kostne a presety domyslne

Najwazniejsze rozroznienie:

- `bone_soft` i `bone_contrast` to presety prezentacji,
- nie sa to presety fizyczne ani materialowe.

Same z siebie:

- nie wlaczaja `piecewise_bone`,
- nie wlaczaja `bone_threshold`,
- nie zmieniaja `output_mode`,
- nie zmieniaja geometrii.

Jesli chcesz uzyskac faktycznie bardziej "kostny" obraz, najlepszy efekt daje
polaczenie:

1. `Physics -> Response = piecewise_bone` albo `bone_threshold`
2. `Presentation -> Preset = bone_soft` albo `bone_contrast`

Wtedy:

- fizyka wzmacnia wklad zakresu kostnego,
- prezentacja dodatkowo ustawia tonalnosc pod ten zakres.

## 9. Adnotacje i aktualizacja obrazu

Po uruchomieniu symulacji:

- surowy wynik trafia do `last_raw_projection`,
- pozycje adnotacji trafiaja do `last_projected_annotations`,
- obraz wyswietlany w workspace jest budowany osobno.

Przycisk `Update display`:

- nie uruchamia projekcji od nowa,
- bierze `last_raw_projection`,
- ponownie stosuje aktualny model prezentacji,
- ponownie rysuje overlaye.

To wygodny sposob na szybkie porownywanie presetow bez liczenia calego RTG od
nowa.

## 10. Rekomendacje praktyczne

### Szybka ocena geometrii

- `quality = draft`
- `presentation = default`
- `output_mode = integral`

### Neutralny podglad roboczy

- `presentation = balanced`
- `Response = linear` lub `piecewise_soft_tissue`

### Podkreslenie kosci bez bardzo agresywnego wygladu

- `Response = piecewise_bone`
- `presentation = bone_soft`

### Maksymalne uwypuklenie struktur kostnych

- `Response = bone_threshold` lub `piecewise_bone`
- `presentation = bone_contrast`

### Wyglad bardziej "filmowy"

- `presentation = film_soft`

## 11. Ograniczenia

- Warstwa prezentacji nie jest modelem klinicznego detektora.
- Presety `bone_*` nie sa kalibracja medyczna, tylko gotowymi ustawieniami
  mapowania tonalnego.
- `film` nie symuluje fizyki kliszy, tylko wyglad zblizony tonalnie.
- Przy `raw` widok w GUI nadal przechodzi przez konwersje do formatu
  wyswietlanego, wiec "surowosc" dotyczy modelu prezentacji, nie bufora ekranu.
