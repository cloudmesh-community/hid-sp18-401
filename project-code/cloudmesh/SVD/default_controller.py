import connexion
import six

from swagger_server.models.svd import SVD  # noqa: E501
from swagger_server import util
from svd_stub import svd_main

def svd_get():  # noqa: E501
    """svd_get

    Display test accuracy of test images on compressed network after applying SVD on the original MNIST network # noqa: E501


    :rtype: SVD
    """
    return SVD(svd_main())
