# extlinks = {
#     # 'duref': ('https://docutils.sourceforge.io/docs/ref/rst/'
#     #           'restructuredtext.html#%s', ''),
#     # 'durole': ('https://docutils.sourceforge.io/docs/ref/rst/'
#     #            'roles.html#%s', ''),
#     # 'dudir': ('https://docutils.sourceforge.io/docs/ref/rst/'
#     #           'directives.html#%s', ''),
#     # 'daobook': ('https://daobook.github.io/%s', ''),
# }

intersphinx_mapping = {
    'python': ('https://daobook.github.io/cpython/', None),
    'sphinx': ('https://daobook.github.io/sphinx/', None),
    'peps': ('https://daobook.github.io/peps', None),
    'pytorch': ('https://pytorch.org/docs/stable', None),
    'pytorchx': ("https://xinetzone.github.io/pytorch-book/api", None),
    'torchvision': ('https://pytorch.org/vision/stable', None),
    'torchtext': ('https://pytorch.org/text/stable/', None),
    'torchaudio': ('https://pytorch.org/audio/stable/', None)
}

# Napoleon 设置
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True