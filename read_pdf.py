import PyPDF2

# Function to read the PDF file
def read_pdf(file_path):
    try:
        # Open the PDF file in read-binary mode
        with open(file_path, 'rb') as pdf_file:
            # Create a PDF Reader object
            reader = PyPDF2.PdfReader(pdf_file)

            # Extract text from each page
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            return text
    except Exception as e:
        return f"An error occurred: {e}"

# Path to the PDF file
pdf_path = rf"D:\Masters_Sub\CMPE_257\project\CMPE-257_Machine Learning_Project_Proposal.pdf"


# Read and print the content of the PDF file
pdf_text = read_pdf(pdf_path)
print(pdf_text)