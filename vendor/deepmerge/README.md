# Vendored `deepmerge`

This is `deepmerge` 4.3.1 (MIT, see `license.txt`) with the fix proposed in
[TehShrike/deepmerge#273](https://github.com/TehShrike/deepmerge/issues/273)
for CVE-2026-93753 / GHSA-gcqx-w48m-w2h4. The upstream package has no released
patch, and the advisory has no fixed version.

Changes from upstream:

- `__proto__`, `constructor` and `prototype` keys are rejected by name in both
  merge loops.
- Keys are written with `Object.defineProperty`, so the `__proto__` accessor is
  never invoked.
- `propertyIsOnObject` fails closed when the `in` check throws.

The version is set to `4.3.2` so version-range scanners stop matching the
`<= 4.3.1` advisory; the root `package.json` pins it with
`"overrides": { "deepmerge": "file:vendor/deepmerge" }`.

Only `@expo/config` (build tooling, via `@expo/cli`) uses this package. Delete
the override and this folder as soon as upstream publishes a real fix.
`npm run check` covers the guard with `scripts/check-deepmerge.mjs`.
