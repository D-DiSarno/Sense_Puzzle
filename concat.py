import subprocess


def concatenate_fasta(input_files, output_file):
    cat_command = ["cat"] + input_files
    with open(output_file, "w") as output_file_handle:
        subprocess.run(cat_command, stdout=output_file_handle)


if __name__ == "__main__":
    # Sostituisci con i nomi dei tuoi file FASTA
    input_files = ["QIITA/out_1.fa", "QIITA/out_2.fa", "QIITA/out_3.fa"]

    # Sostituisci con il nome desiderato per il file di output
    output_file = "converted/merged.fa"

    concatenate_fasta(input_files, output_file)
    print(f"I file sono stati concatenati con successo. Il risultato è salvato in '{output_file}'.")
