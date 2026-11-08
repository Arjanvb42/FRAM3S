# FRAM3

Van video of foto's naar een 3D-model, dat je daarna bewerkt en exporteert. Het bewerken werkt zo eenvoudig als in Tinkercad, op een printbed zoals in OrcaSlicer. Alles draait lokaal op je eigen computer.

## Starten

```bash
./fram3.sh
```

De eerste keer maakt het script een Python-omgeving aan in `.venv` en installeert het de pakketten. Daarna opent de browser op http://127.0.0.1:8733.

Voor scannen zijn COLMAP en OpenMVS nodig. Die installeer je eenmalig met:

```bash
scripts/install-reconstruction.sh
```

Het script bouwt beide tools met CUDA als de NVIDIA-driver en de CUDA-toolkit aanwezig zijn, en anders voor de CPU. Het bouwen duurt 20 tot 40 minuten. Rechtsboven in de app zie je welke tools gevonden zijn.

## Wat het kan

**Scannen**
- Sleep video's en/of foto's in het tabblad Scannen.
- FRAM3 kiest uit elke video de scherpste frames, berekent de cameraposities (COLMAP) en maakt daarna een dichte puntenwolk, een mesh en een textuur (OpenMVS).
- Losse stukjes worden verwijderd en het model komt op het bed in de editor.
- Een scan heeft geen vaste schaal. Stel daarom de echte maat in bij Afmeting (B/D/H).

**Bewerken**
- Basisvormen: kubus, cilinder, bol, kegel, piramide, torus, wig, buis en halve bol. Klik erop of sleep ze naar het bed.
- Elk object kan Massief of Gat zijn. Samenvoegen (Ctrl+G) telt de massieve objecten bij elkaar op en snijdt de gaten eruit. Dat gebeurt exact en het resultaat is altijd waterdicht.
- Verplaatsen, draaien en schalen met de gizmo (W/E/R), of met exacte maten in mm. Je kunt de verhouding vastzetten, en alles snapt op 1 mm en 15°.
- Verder: op bed leggen (D), dupliceren (Ctrl+D), spiegelen, verbergen, ongedaan maken en opnieuw doen (Ctrl+Z en Ctrl+Y).
- Repareren voor printen: losse stukjes en de ondergrond verwijderen, gaten dichten, waterdicht maken, vereenvoudigen en schalen naar een maat in mm.

**Bestandsformaten**

| Formaat | Import | Export | Opmerking |
|---|---|---|---|
| STL | ✓ | ✓ | één mesh |
| 3MF | ✓ | ✓ | losse objecten en kleuren, voor slicers |
| OBJ | ✓ | ✓ | |
| GLB / glTF | ✓ | ✓ | omgerekend naar Y-up en meters |
| PLY, OFF | ✓ | ✓ | |
| Collada (DAE) | ✓ | ✓ | |
| FBX, USD/USDZ, Alembic, X3D | ✓ | ✓ | via Blender (wordt automatisch gevonden, ook de Flatpak-versie) |
| STEP | ✓ | – | via `cascadio` |

Intern rekent FRAM3 net als slicers in Z-up en millimeters.

## Structuur

```
fram3/formats.py      import/export van alle formaten
fram3/repair.py       opschonen, waterdicht maken, exacte booleans (manifold3d)
fram3/reconstruct.py  scanpipeline: frames → COLMAP → OpenMVS → opschonen
fram3/server.py       lokale server (FastAPI) + API
web/                  editor (three.js, zonder build-stap)
scripts/              installatie van de scantools
```

## Instellingen (omgevingsvariabelen of ~/.config/fram3/env)

- `FRAM3_COLMAP`: pad naar het `colmap`-programma
- `FRAM3_OPENMVS_DIR`: map met `DensifyPointCloud`, `ReconstructMesh` enzovoort
- `FRAM3_BLENDER`: Blender-commando, als het niet automatisch gevonden wordt
- `FRAM3_DATA`: waar scans worden opgeslagen (standaard `~/.local/share/fram3`)

## Licenties

FRAM3 gebruikt alleen onderdelen die commercieel gebruik toestaan: three.js (MIT), three-bvh-csg (MIT), trimesh (MIT), manifold3d (Apache-2.0), FastAPI (MIT), COLMAP (BSD) en OpenMVS (AGPL, als losse tool aangeroepen). Laat de AGPL-voorwaarden van OpenMVS nog controleren voordat je FRAM3 verkoopt.
