from .base import MediaItem, Page, Source  # noqa: F401
from .commons import CommonsSource  # noqa: F401
from .local import FolderSource, UploadSource, UrlSource  # noqa: F401
from .zenodo import ZenodoSource  # noqa: F401


def default_sources(folders=None, zenodo=True, commons=True):
    out = [UrlSource(), UploadSource()]
    if folders:
        out.append(FolderSource(folders))
    if zenodo:
        out.append(ZenodoSource())
    if commons:
        out.append(CommonsSource())
    return out
