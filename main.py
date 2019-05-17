import sys

from PyQt5.QtWidgets import QApplication

from LACV.controller import LACVController
from LACV.widgets import LACVWindow

import matplotlib
matplotlib.use('Qt5Agg')

lacv_controller = LACVController()

if __name__ == "__main__":
    app = QApplication(sys.argv)    
    lacv_window = LACVWindow(lacv_controller)
    lacv_window.show()
    sys.exit(app.exec_())
