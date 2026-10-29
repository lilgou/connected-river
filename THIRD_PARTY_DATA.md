# Data sources and licences

The code in this repository has an MIT licence. The datasets keep their own
licences, so these should be checked before sharing downloaded data or outputs.

## Data downloaded by the workflow

- **Environment Agency data** provides the chemistry, ecology and wider river
  context used in the notebooks. The source links are listed in
  `config/source_catalog.yaml` and are also recorded when the workflow runs.
- **National River Flow Archive (NRFA)** data provides the station catalogue
  and gauged daily flows. Reuse should include the acknowledgement “Data from
  the UK National River Flow Archive”.
- **ChemPop version 2** is downloaded only for the historical continuation at
  the end of The Connected River (Part 2) - Exploring. It is available under the Open Government Licence
  v3.0. Cite: Bachiller-Jareno, N. et al. (2025), *Macroinvertebrate taxonomic
  abundance, water quality, river flow, air temperature and environmental site
  descriptors from English rivers, 1965–2018, version 2.0*. NERC EDS
  Environmental Information Data Centre. DOI
  `10.5285/23f194b0-1e21-4f14-be9c-6eabdcc1feb7`.

## Data downloaded separately

The topology-enabled UKCEH river network used here is not included. It must be
requested through the UKCEH data-licensing route and supplied as the CSV
described in The Connected River (Part 1) - Building. Cite: Moore, R.V., Morris,
D.G. and Flavin, R.W. (1994), *Sub-set of UK digital 1:50,000 scale river
centre-line network*.

- [UKCEH river-network access and licensing](https://www.ceh.ac.uk/data/15000-watercourse-network)

The hydrometric-area boundaries are downloaded separately from the UKCEH
catalogue. Cite: National River Flow Archive (2014), *Hydrometric Areas for
Great Britain and Northern Ireland with Coastline*, DOI
`10.5285/1957166d-7523-44f4-b279-aa5314163237`.

- [Hydrometric Areas with Coastline](https://catalogue.ceh.ac.uk/documents/1957166d-7523-44f4-b279-aa5314163237)

The discharge-site layer comes from The Rivers Trust public ArcGIS service and
is derived from the Environment Agency register. Check the licence attached to
the source before redistributing identifiable outlet records.

Raw downloads, the licensed river network and the generated joined dataset are
not included in this repository. Anyone sharing additional outputs should first
check the terms for the sources included in those outputs.
