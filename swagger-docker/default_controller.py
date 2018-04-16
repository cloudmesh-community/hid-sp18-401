import connexion
import six

from swagger_server.models.cv import CV  # noqa: E501
from swagger_server import util
from cv import get_cross_validation_score

def cv_get():  # noqa: E501
    """cv_get

    Returns cross validation average mean squared accuracy of linear regression model built on the &#39;Advertising&#39; dataset # noqa: E501


    :rtype: CV
    """
    return CV(get_cross_validation_score())
