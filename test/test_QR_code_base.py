import sys
sys.path.append('../')

from src.QR_code_base_gen import *

def print_qr(qrcode: QrCode) -> None:
	"""Prints the given QrCode object to the console."""
	border = 4
	print(qrcode.get_size())
	for y in range(-border, qrcode.get_size() + border):
		for x in range(-border, qrcode.get_size() + border):
			print("\u2588 "[1 if qrcode.get_module(x,y) else 0] * 2, end="")
		print()
	print()

def do_basic_demo() -> None:
	"""Creates a single QR Code, then prints it to the console."""
	text = "www.baidu.com"      # User-supplied Unicode text
	errcorlvl = QrCode.Ecc.LOW  # Error correction level
	
	# Make and print the QR Code symbol
	qr = QrCode.encode_text(text, errcorlvl)
	print_qr(qr)
	
def main():
	do_basic_demo()
	
if __name__ == "__main__":
    main()
