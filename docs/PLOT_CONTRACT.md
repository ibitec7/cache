# Plot and typesetting contract

`src/mrc/plots/style.py` is the single source of truth. Every figure script calls
`apply_style()` first and sets nothing else.

| Rule | Value | Reason |
|---|---|---|
| Font family | serif | presentation is printed and projected |
| Serif list | `Nimbus Roman`, `Liberation Serif`, `Times New Roman`, `DejaVu Serif` | Times New Roman is not installed on this machine; Nimbus Roman is metrically identical, so line breaks and widths match a TNR document |
| Maths font | STIX | matches `newtxmath` in the report body |
| Single column width | 3.35 in | fits a 6.5 in text block at two-up |
| Double column width | 7.0 in | full text width |
| Output | PDF, 300 dpi, `bbox_inches="tight"`, `pdf.fonttype=42` | vector, editable, no font substitution at the printer |
| Grid | off | the data is a curve, gridlines add ink without information |
| Spines | top and right removed | same |
| Ticks | out, 8 pt | less visual noise than inward ticks |
| Legend | no frame, at most 4 entries | more than 4 series means the figure is doing two jobs |
| Annotations | at most one per panel | an annotation is an argument, so make only one |
| Titles | none inside the figure | the caption states the claim; the axis labels state the quantity |
| Axis labels | `Reuse distance`, `Miss ratio`, `Cache capacity (objects)` | plain words, unit in parentheses only when digits need it |
| Colours | fixed palette in `style.py` | the same series keeps the same colour across every figure in the report |

## Enforcement

`tests/test_smoke.py::test_style_contract` asserts the font list, grid state and
spines after `apply_style()`. A figure that needs a different treatment adds a
function to `style.py`, not a local `rcParams` edit.

## Report typesetting

- Template inherited from the course Overleaf project: `article`, 10 pt, A4, manual
  margin adjustment, `unsrt` bibliography, `sections/` per section, appendix after
  references.
- `newtxtext,newtxmath` added so the body is Times-compatible with the figures.
- Main body 8 to 10 pages. References. Then appendix, unlimited.
- `booktabs` tables only, no vertical rules, units in the column header.
