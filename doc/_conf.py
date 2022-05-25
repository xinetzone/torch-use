# 项目信息
project = 'torch-book'
copyright = '2021, xinetzone'
author = 'xinetzone'
html_baseurl = 'https://xinetzone.github.io/torch-book'

comments_config = {
    "hypothesis": True,
    "dokieli": False,
    "utterances": {
        "repo": "xinetzone/torch-book",
        "optional": "config",
    }
}

# -- 主题设置 -------------------------------------------------------------------
# 定制主侧栏
html_sidebars = {
    "*" : [
        "sidebar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ],
    "posts/**": [
        "sidebar-logo.html",
        "postcard.html",
        "recentposts.html",
        "tagcloud.html",
        "categories.html",
        "archives.html",
        "searchbox.html",
    ],
}

extra_navbar = """<div>
版权所有 © 2021 <a href="https://xinetzone.github.io/">xinetzone</a></div>
<div>由 <a href="https://ebp.jupyterbook.org/">EBP</a> 提供技术支持</div>
<a href="https://torch-book.readthedocs.io/zh/latest/">版本切换</a>
"""
autosummary_generate = True

html_theme_options = {
    # -- 如果你的文档只有一个页面，而且你不需要左边的导航栏，那么 ---------------
    # 你可以在 单页模式 下运行，
    # "single_page": False,  # 默认 `False`
    # 默认情况下，编辑按钮将指向版本库的根。
    # 如果你的文档被托管在一个子文件夹中，请使用以下配置：
    "path_to_docs": "docs/",  # 文档的路径，默认 `docs/``
    "repository_url": "https://github.com/xinetzone/torch-book",
    "repository_branch": "main",  # 文档库的分支，默认 `master`
    # -- 在导航栏添加一个按钮，链接到版本库的议题 ------------------------------
    # （与 `repository_url` 和 `repository_branch` 一起使用）
    "use_issues_button": True,  # 默认 `False`
    # -- 在导航栏添加一个按钮，以下载页面的源文件。
    "use_download_button": True,  # 默认 `True`
    # 你可以在每个页面添加一个按钮，允许用户直接编辑页面文本，
    # 并提交拉动请求以更新文档。
    "use_edit_page_button": True,
    # 在导航栏添加一个按钮来切换全屏的模式。
    "use_fullscreen_button": True,  # 默认 `True`
    # -- 在导航栏中添加一个链接到文档库的按钮。----------------------------------
    "use_repository_button": True,  # 默认 `False`
    # -- 包含从 Jupyter 笔记本建立页面的 Binder 启动按钮。 ---------------------
    # "launch_buttons": '', # 默认 `False`
    "home_page_in_toc": False,  # 是否将主页放在导航栏（顶部）
    # -- 只显示标识，不显示 `html_title`，如果它存在的话。-----
    "logo_only": True,
    # -- 在导航栏中显示子目录，向下到这里列出的深度。 ----
    # "show_navbar_depth": 2,
    # -- 在侧边栏页脚添加额外的 HTML -------------------
    # （如果 `sbt-sidebar-footer.html `在 `html_sidebars` 中被使用）。
    "extra_navbar": extra_navbar,
    # -- 在每个页面的页脚添加额外的 HTML。---
    # "extra_footer": '',
    # （仅限开发人员）触发一些功能，使开发主题更容易。
    # "theme_dev_mode": False
    # 重命名页内目录名称
    "toc_title": "导航",
    "launch_buttons": {
        # https://mybinder.org/v2/gh/xinetzone/torch-book/main
        "binderhub_url": "https://mybinder.org",
        # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
        "colab_url": "https://colab.research.google.com/",
        # 你可以控制有人点击启动按钮时打开的界面。
        "notebook_interface": "jupyterlab",
        # "thebe": True,  # Thebe 实时代码单元格
    },
}
# -- 自定义网站的标志 --------------
html_logo = '_static/images/logo.jpg'
# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/images/favicon.jpg"

# -- 自定义网站的标题 --------------
# html_title = '动手学习 Python'

# ========== ABlog 配置 ============================================================
blog_path = "posts"
blog_post_pattern = "posts/*.md"
blog_baseurl = "https://xinetzone.github.io/torch-book"
fontawesome_included = True
post_auto_image = 1
post_auto_excerpt = 2
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
# --    博客作者、语言和位置 -------------------------------------------------

# 一个作者名字的字典，映射到作者的完整显示名称和链接。
# 字典的键值应该在 ``post`` 指令中使用，以指代作者。默认是 ``{}``。
blog_authors = {
    "lxw": ("刘新伟", None),
}

# 语言代码名称的字典，映射到这些语言的完整显示名称和链接。
# 类似于 :confval:`blog_authors`，
# 字典的键应该在 `post` 指令中使用，以指代位置。默认是 `{}`。
blog_languages = {'zh': ('Chinese', None), 'en': ('English', None)}

# 默认作者的名字
blog_default_author = "lxw"
# 默认语言的代码名称
blog_default_language = 'zh'
# 在 blog_locations 中定义的默认位置的名称。
# blog_default_location = None

# -- 博客帖子相关 --------------------------------------------------------

# 帖子的日期格式。默认 ``'%b %d, %Y'``
#  ``datetime.date.strftime()`` 的参数
post_date_format = '%c'
post_date_format_short = '%b %d, %Y'

