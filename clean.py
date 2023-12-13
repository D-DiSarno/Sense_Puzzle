def fix_fasta_format(input_file, output_file):
    with open(input_file, "r") as input_f, open(output_file, "w") as output_f:
        for line in input_f:
            if line.startswith(">"):
                # Modifica l'intestazione e scrivi nel nuovo file
                header_parts = line.strip().split(" ")
                new_header = f"{header_parts[0]}#{header_parts[-1][:-1]}"
                output_f.write(new_header + "\n")
            else:
                # Scrivi la sequenza su una nuova linea nel nuovo file
                output_f.write(line.strip() + "\n")
if __name__ == "__main__":
    input_file_path = "converted/Unified.fa"  # Sostituisci con il percorso corretto del tuo file FASTA
    output_file_path = "clean.fa"   #Sostituisci con il percorso desiderato per il nuovo file

    fix_fasta_format(input_file_path, output_file_path)
