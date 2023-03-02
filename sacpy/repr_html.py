# This code is from  Lifei Lin (Sun Yat-sen University) 
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import read_binary
import xarray as xr
from xarray.core.formatting import inline_variable_array_repr, short_data_repr, format_array_flat
from xarray.core.options import _get_boolean_with_default
import numpy as np

STATIC_FILES = (
    ("xarray.static.html", "icons-svg-inline.html"),
    ("xarray.static.css", "style.css"),
)

def res_repr_html(arr):
    arr_name = arr.name
    header_components = [
        f"<h3>{arr_name}</h3>",
        # f"<div class='xr-obj-type'>{obj_type}</div>",
        # f"<div class='xr-array-name'>{arr_name}</div>",
    ]
    res_dict = arr.__dict__.copy()
    sections = [attribute_section(res_dict)]
    return _obj_repr(arr, header_components, sections)

@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    return [
        read_binary(package, resource).decode("utf-8")
        for package, resource in STATIC_FILES
    ]

def _obj_repr(obj, header_components, sections):
    """Return HTML repr of an xarray object.

    If CSS is not injected (untrusted notebook), fallback to the plain text repr.

    """
    header = f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='xr-section-item'>{s}</li>" for s in sections)

    icons_svg, css_style = _load_static_files()
    return (
        "<div>"
        f"{icons_svg}<style>{css_style}</style>"
        f"<pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre>"
        "<div class='xr-wrap' hidden>"
        f"{header}"
        f"<ul class='xr-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )

def collapsible_section(
    name, inline_details="", details="", n_items=None, enabled=True, collapsed=False
):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
    enabled = "" if enabled and has_items else "disabled"
    collapsed = "" if collapsed or not has_items else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    return (
        f"<input id='{data_id}' class='xr-section-summary-in' "
        f"type='checkbox' {enabled} {collapsed}>"
        f"<label for='{data_id}' class='xr-section-summary' {tip}>"
        f"{name}:{n_items_span}</label>"
        f"<div class='xr-section-inline-details'>{inline_details}</div>"
        f"<div class='xr-section-details'>{details}</div>"
    )

def _mapping_section(
    mapping, name, details_func, max_items_collapse, expand_option_name, enabled=True
):
    n_items = len(mapping)
    expanded = _get_boolean_with_default(
        expand_option_name, n_items < max_items_collapse
    )
    collapsed = not expanded

    return collapsible_section(
        name,
        details=details_func(mapping),
        n_items=n_items,
        enabled=enabled,
        collapsed=collapsed,
    )

def short_data_repr_html(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if hasattr(internal_data, "_repr_html_"):
        return internal_data._repr_html_()
    text = escape(short_data_repr(array))
    return f"<pre>{text}</pre>"

def _icon(icon_name):
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return (
        "<svg class='icon xr-{0}'>"
        "<use xlink:href='#{0}'>"
        "</use>"
        "</svg>".format(icon_name)
    )

def summarize_attrs(name, var, is_index=False, dtype=None, preview=None):
    if hasattr(var,"variable"):
      variable = var.variable
      dims_str = f"({', '.join(dim for dim in var.dims)})"
      dtype = dtype or escape(str(variable.dtype))
      data_repr = var._repr_html_()
    else:
      isDict = type(var) == dict
      isStr = type(var) == str
      isList = type(var) == list
      variable = xr.DataArray(var)
      dims = np.shape(variable)
      lens = len(dims)
      if(lens == 0): dims_str = ""
      elif(lens == 1): dims_str = dims[0]
      else: dims_str = dims
      dtype = dtype or escape(str(variable.dtype))
      if(isDict): dtype = "dict"
      elif(isStr): dtype = "str"
      data_repr = f"<p></p><ul class='xr-var-list'>{var}</ul><p></p>"

    disabled = "disabled"

      
    cssclass_idx = " class='xr-has-index'" if is_index else ""
    name = escape(str(name))
    # dtype = "dtype_test"

    # "unique" ids required to expand/collapse subsections
    attrs_id = "attrs-" + str(uuid.uuid4())
    data_id = "data-" + str(uuid.uuid4())

    preview = preview or escape(format_array_flat(variable, 35))

    attrs_icon = _icon("icon-file-text2")
    data_icon = _icon("icon-database")
    return (
        f"<div class='xr-var-name'><span{cssclass_idx}>{name}</span></div>"
        f"<div class='xr-var-dims'>{dtype}</div>"
        f"<div class='xr-var-dtype'>{dims_str}</div>"
        f"<div class='xr-var-preview xr-preview'>{preview}</div>"
        f"<input id='{attrs_id}' class='xr-var-attrs-in' "
        f"type='checkbox' {disabled}>"
        f"<label for='{attrs_id}' title='Show/Hide attributes'>"
        f"{attrs_icon}</label>"
        f"<input id='{data_id}' class='xr-var-data-in' type='checkbox'>"
        f"<label for='{data_id}' title='Show/Hide data repr'>"
        f"{data_icon}</label>"
        # f"<div class='xr-var-attrs'>{attrs_ul}</div>"
        f"<div class='xr-var-data'>{data_repr}</div>"
    )

def summarize_vars(dict):

    vars_li = "".join(
        f"<li class='xr-var-item'>{summarize_attrs(k, v)}</li>"
        for k, v in zip(dict, dict.values())
    )

    return f"<ul class='xr-var-list'>{vars_li}</ul>"

attribute_section = partial(
    _mapping_section,
    name="Attributes",
    details_func=summarize_vars,
    max_items_collapse=15,
    expand_option_name="display_expand_data_vars",
)