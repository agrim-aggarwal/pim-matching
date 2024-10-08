import os


def main():
	cwd = os.getcwd()
	text = 'hello world from custom file'
	
	with open(f'{cwd}/file_generated_from_code.txt', 'wb') as fp:
		fp.write(text.encode('utf-8'), fp)
	

if __name__ == '__main__':
	main()