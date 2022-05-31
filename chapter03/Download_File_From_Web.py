import urllib.request
import zipfile


class Download_File_From_Web():

	def __init__(self, url, filename):
		self.url = url
		self.filename = filename
	pass

	def download_data_to(self, folder):

		urllib.request.urlretrieve(self.url, self.filename)
		self.__unzip_file(folder)
	pass

	def __unzip_file(self, folder):
		zip_ref = zipfile.ZipFile(self.filename, 'r')
		zip_ref.extractall(folder)
		zip_ref.close()
	pass

pass