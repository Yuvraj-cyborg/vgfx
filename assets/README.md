# assets

Third-party 3D assets vendored here (not code).

- `ogre.obj` — "Ogre" test model from
  <https://github.com/alecjacobson/common-3d-test-models>
  (originally the MPI-I Ogre). No clear license file ships with that repo,
  but the model is a decades-old public test mesh; treated as
  public-domain test data. If you ship a product, swap in a model with an
  explicit license (e.g. CC0 Quaternius packs from <https://quaternius.com>).
- The file is positions + triangle faces only (no normals/UVs/materials),
  which is exactly what `Mesh::load_obj` expects to demonstrate: normals
  are computed on load.
