================================================
HistomicsTK-deeplab
================================================

HistomicsTK-deeplab is a modified version of `HistomicsTK`_ by `Brendon Lutnick`_. This code has been modified to include native WSI segmentation via Google's `DeepLab`_ v3+ architecture. The `DeepLab codebase`_ has also been modified to directly and efficiently train and predict on histology Whole Slide Images (WSIs). The `DeepLab codebase`_ can be run via the command line, within this docker container (which will install all the needed dependencies), or via the options below.

`HistomicsTK`_ is a Python package for the analysis of digital pathology images. It can function as a stand-alone library, or as a Digital Slide Archive plugin that allows users to invoke image analysis jobs through HistomicsUI. The functionality offered by HistomicsTK can be extended using `slicer cli web <https://github.com/girder/slicer_cli_web>`__ which allows developers to integrate their image analysis algorithms into DSA for dissemination through HistomicsUI.

HistomicsTK can be used in two ways:

- **As a pure Python package**: enables application of image analysis algorithms to data independent of the `Digital Slide Archive`_ (DSA). HistomicsTK provides a collection of fundamental algorithms for tasks such as color normalization, color deconvolution, nuclei segmentation, and feature extraction. Read more about these capabilities here:  `api-docs <https://digitalslidearchive.github.io/HistomicsTK/api-docs.html>`__ and `examples <https://digitalslidearchive.github.io/HistomicsTK/examples.html>`__ for more information. The HistomicsTK-DeepLab specific plugins are documented `here <https://buffalo.app.box.com/s/3d56aoasjcwryw9ktyahhlzm8skl8c9b>`__, and available for testing on our `test server <https://athena.ccr.buffalo.edu/>`__.

  **Installation instructions on Linux:**

  *To install HistomicsTK using PyPI*::

  $ python -m pip install histomicstk

  *To install HistomicsTK from source*::

  $ git clone https://github.com/DigitalSlideArchive/HistomicsTK/
  $ cd HistomicsTK/
  $ python -m pip install setuptools-scm Cython>=0.25.2 scikit-build>=0.8.1 cmake>=0.6.0 numpy>=1.12.1
  $ python -m pip install -e .

  HistomicsTK uses the `large_image`_ library to read content from whole-slide and microscopy image formats. Depending on your exact system, installing the necessary libraries to support these formats can be complex.  There are some non-official prebuilt libraries available for Linux that can be included as part of the installation by specifying ``pip install histomicstk --find-links https://girder.github.io/large_image_wheels``. Note that if you previously installed HistomicsTK or large_image without these, you may need to add ``--force-reinstall --no-cache-dir`` to the ``pip install`` command to force it to use the find-links option.

  The system version of various libraries are used if the ``--find-links`` option is not specified.  You will need to use your package manager to install appropriate libraries (on Ubuntu, for instance, you'll need ``libopenslide-dev`` and ``libtiff-dev``).

  **To install from source on Windows**:

  1- Run the following::

  $ pip install large-image
  $ pip install cmake
  $ git clone https://github.com/DigitalSlideArchive/HistomicsTK/
  $ cd HistomicsTK/
  $ python -m pip install setuptools-scm Cython>=0.25.2 scikit-build>=0.8.1 cmake>=0.6.0 numpy>=1.12.1

  2- Run ``pip install libtiff``

  3- Replace ``large-image[sources]`` with ``large-image[pil,tiff]`` in ``setup.py``.

  4- Install Visual Studio 15 2017 `Community Version <https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads>`_

  5- Install C++ build tools. Under Tools > Get Tools and Features ... > Desktop Development with C++, ensure that the first 8 boxes are checked.

  6- Run this::

  $ python -m pip install -e .
  $ pip install girder-client


- **As a image-processing task library for HistomicsUI and the Digital Slide Archive**: This allows end users to apply containerized analysis modules/pipelines over the web. See the `Digital Slide Archive`_ for installation instructions.

Refer to the `DSA website`_ for more information.

See Also
---------

**DSA/HistomicsTK project website:**
`Demos <https://digitalslidearchive.github.io/digital_slide_archive/demos-examples/>`_ |
`Success stories <https://digitalslidearchive.github.io/digital_slide_archive/success-stories/>`_

**Source repositories:** `Digital Slide Archive`_ | `HistomicsUI`_ | `large_image`_ | `slicer_cli_web`_

**Discussion:** `Discourse forum`_ | `Gitter Chatroom`_

.. Links for everything above (not rendered):
.. _Brendon Lutnick: https://github.com/brendonlutnick
.. _HistomicsTK: https://github.com/DigitalSlideArchive/HistomicsTK
.. _DeepLab: https://github.com/tensorflow/models/tree/master/research/deeplab
.. _DeepLab codebase: https://github.com/SarderLab/HistomicsTK-deeplab/tree/main/histomicstk/deeplab
.. _Digital Slide Archive: http://github.com/DigitalSlideArchive/digital_slide_archive
.. _HistomicsUI: http://github.com/DigitalSlideArchive/HistomicsUI
.. _large_image: https://github.com/girder/large_image
.. _DSA website: https://digitalslidearchive.github.io/digital_slide_archive/
.. _slicer execution model: https://www.slicer.org/slicerWiki/index.php/Slicer3:Execution_Model_Documentation
.. _slicer_cli_web: https://github.com/girder/slicer_cli_web
.. _Docker: https://www.docker.com/
.. _Kitware: http://www.kitware.com/
