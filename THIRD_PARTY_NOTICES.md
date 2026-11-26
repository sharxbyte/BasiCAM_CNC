# Third-party software in BasiCAM CNC

BasiCAM CNC is licensed under the GNU General Public License v3.0 or later (`LICENSE`). The Windows
build bundles the components below, each under its own license. Their license files are included in
the build folder next to each package where the package ships one. The source code of each is
available from its project.

| Component | License | Project |
|---|---|---|
| Python | PSF-2.0 | https://www.python.org |
| PySide6 / Shiboken6 (Qt for Python) and Qt | LGPL-3.0 (used under LGPL-3.0) | https://www.qt.io/qt-for-python |
| VTK | BSD-3-Clause | https://vtk.org |
| PyVista, pyvistaqt, pyvista-cad | MIT | https://pyvista.org |
| cadquery-ocp (OCP) | Apache-2.0 | https://github.com/CadQuery/OCP |
| Open CASCADE Technology (inside OCP) | LGPL-2.1 with the OCCT exception | https://dev.opencascade.org |
| python-solvespace (SolveSpace solver) | GPL-3.0-or-later | https://github.com/KmolYuan/solvespace |
| NumPy | BSD-3-Clause (and bundled 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | BSD-3-Clause | https://scipy.org |
| Shapely (and GEOS) | BSD-3-Clause (GEOS: LGPL-2.1) | https://shapely.readthedocs.io |
| trimesh | MIT | https://trimesh.org |
| ezdxf | MIT | https://ezdxf.mozman.at |
| pySerial | BSD-3-Clause | https://github.com/pyserial/pyserial |
| pywin32 | PSF-2.0 | https://github.com/mhammond/pywin32 |
| Matplotlib | PSF-based Matplotlib license | https://matplotlib.org |
| Pillow | MIT-CMU | https://python-pillow.org |
| pooch, scooby, qtpy, platformdirs, pyparsing, fonttools | BSD-3-Clause / MIT | see each project |

The Qt and Open CASCADE libraries are shipped as separate DLLs in the build folder, so they can be
replaced with compatible versions, as the LGPL allows.

The Windows build is compiled with Nuitka (https://nuitka.net, Apache-2.0); its runtime parts are
covered by Nuitka's license exception for compiled programs.
