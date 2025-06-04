import importlib
import pkgutil

# Import all services
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    print("Importing", module_name)
    importlib.import_module(module_name)
