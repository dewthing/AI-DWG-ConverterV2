# Input drawings for GitHub Actions

Put the image or PDF that should be converted in this folder, commit it to a
private repository, then run the Convert CAD drawing workflow from the Actions
tab. Use a private repository for drawings that are confidential.

The workflow accepts a path below input/ only and creates downloadable DXF,
quality-preview PNG, JSON report, and ZIP artifacts. It does not create DWG,
because the runner does not include a licensed DWG converter.

