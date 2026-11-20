# biospeak

A Python package that enables biocircuit designers to 'speak' about biomolecular systems by abstracting chemical reaction networks into their input-output interfaces --- b-i/o-speak! 

## Installation

From this folder:

```bash
pip install -e ".[examples]"
```

## Quick Start

```python
from biospeak import enzyme_contract

contract = enzyme_contract(K=20, kcat=3, regime="substrate")
print(contract)
```

## Examples

### Canonical CRN Examples

* [Single protein expression](examples/canonical/single_protein_expression.ipynb)
* [Composition of multiple protein expression subsystems](examples/canonical/protein_expression_composition.ipynb)
* [Enzymatic Output Flux](examples/canonical/enzymatic_output_flux.ipynb)

### Biocircuit Design Applications

* [Library of Logic Gate Contracts Using Cello](examples/biocircuit_design/cello_gate_library.ipynb)
* [Mechanistic Design Space Exploration for an AND Gate](examples/biocircuit_design/and_gate_output_flux.ipynb)
* [RBS library screening for bio-design](examples/biocircuit_design/rbs_library_screening.ipynb)

## Paper

Ayush Pandey (2026). *Exponent Polytopes for Abstracting Chemical Reaction Networks into Formal Specification Models for Compositional Design of Biomolecular Circuits*. bioRxiv. [doi:10.64898/2026.11.13.766398](https://www.biorxiv.org/content/10.64898/2026.11.13.766398v1).

## Citation

```bibtex
@article{pandey2026exponent,
  author = {Pandey, Ayush},
  title = {{Exponent Polytopes for Abstracting Chemical Reaction Networks into Formal Specification Models for Compositional Design of Biomolecular Circuits}},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.11.13.766398},
  url = {https://www.biorxiv.org/content/10.64898/2026.11.13.766398v1}
}
```

## Contact

[Ayush Pandey](https://ayush-pandey.github.io/)

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).
