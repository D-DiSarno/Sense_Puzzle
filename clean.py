from Bio import SeqIO
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



def remove_short_and_sequences_with_N(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                seq_line = lines[i + 1].strip()
                if "N" in seq_line or len(seq_line) < 15:
                    i += 2  # Ignora la sequenza e il rigo successivo
                else:
                    outfile.write(lines[i])
                    i += 1
            else:
                outfile.write(lines[i])
                i += 1

def count_nucleotides(file_path):
    sequences = SeqIO.parse(file_path, 'fasta')
    total_count = 0

    for sequence in sequences:
        #nucleotide_count = len(sequence.seq)
        total_count += 1

    return total_count


if __name__ == "__main__":
    input_file_path = "converted/seqs.fa"  # Sostituisci con il percorso corretto del tuo file FASTA
    output_file_path = "converted/seqs_0.fa"   #Sostituisci con il percorso desiderato per il nuovo file

    remove_short_and_sequences_with_N(input_file_path, output_file_path)
    print("FATTO")
    final_count_1 = count_nucleotides(input_file_path)
    print(f"Conteggio iniziale delle sequenze nucleotidiche: {final_count_1}")
    final_count = count_nucleotides(output_file_path)

    print(f"Conteggio finale delle sequenze nucleotidiche: {final_count}")