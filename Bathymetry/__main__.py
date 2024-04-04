import logging

from Bathymetry import bathymetry

logging.basicConfig(level=logging.INFO)


def main():
    logging.info(bathymetry)


if __name__ == '__main__':
    logging.debug('>>> Estamos comenzando la ejecución del paquete.')

    main()

    logging.debug('>>> Estamos finalizando la ejecución del paquete.')