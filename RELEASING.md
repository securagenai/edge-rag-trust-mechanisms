# Releasing a new version

Each GitHub release is archived automatically on Zenodo (all versions: [10.5281/zenodo.23620708](https://doi.org/10.5281/zenodo.23620708)).

1. Update the version and `date-released` in `.zenodo.json` and `CITATION.cff`, and the licence notes in `README.md` if they change.
2. Check that `bash analysis/reproduce_all.sh` still rewrites the released result files with identical values.
3. Commit, then create a GitHub release with tag `vX.Y.Z` and release notes that state the changes since the previous version and the licences.
4. On Zenodo, open the new version created from the release. Zenodo takes a single licence from `.zenodo.json` (MIT), so add the second one: **Edit → Licenses → Add standard → Creative Commons Attribution Share Alike 4.0 International → Publish**. Check that the published record lists both licences.
