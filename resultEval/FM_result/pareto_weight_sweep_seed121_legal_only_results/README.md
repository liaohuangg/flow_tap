# Legal-only Pareto weight sweep results

- Fixed sampling seed: 121
- 12 benchmark cases, 50 thermal/wirelength weight combinations per case
- Included layouts require `legal_report["after"]["is_legal"] == true`
- 493 legal layouts; each JSON has a matching legalization comparison PNG
- `result.csv` contains HotSpot temperature and TAP-2.5D wirelength metrics
- File names encode the thermal and wirelength guidance weights
