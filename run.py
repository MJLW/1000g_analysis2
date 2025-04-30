import argparse

import polars as pl
import numpy as np
import gff3_parser

from os import path
import pdb
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse 1000Genomes RescueRanger and Splice Junction extractions.")
    parser.add_argument("--gff3", help="GFF3 file containing all possible transcripts.")
    parser.add_argument("--analysis_pairs_file", help="File containing tab separated pairs of RescueRanger and Splice Junction extraction files.")

    return parser.parse_args()

def read_gff3(file: str, verbose = True):
    if verbose:
        logging.info(f"Reading in GFF3 file: {file}...")
    return gff3_parser.parse_gff3(file, parse_attributes=True, verbose = False)

def read_file_pairs(file: str, verbose = True):
    if verbose:
        logging.info(f"Reading file pairs...")
    return pl.read_csv(file, separator="\t")


def main():
    args = parse_args()

    gff = read_gff3(args.gff3)
    df_gff = pl.from_pandas(gff).with_columns(pl.col("Start").cast(pl.Int64), pl.col("End").cast(pl.Int64))
    file_pairs = read_file_pairs(args.analysis_pairs_file)

    junction_schema = {
        "chr": pl.Utf8, "start": pl.Int64, "end": pl.Int64, 
        "strand": pl.Int8, "motif": pl.Int8, "annotated": pl.Int8, 
        "unique_junction_reads": pl.Int32, "multi_mapped_junction_reads": pl.Int32, "max_spliced_splice_overhang": pl.Int32
    }

    output = []
    for i, (tab_vcf, junctions) in enumerate(file_pairs.iter_rows()):
        logging.info(f"Collecting read support for pair {i+1}/{file_pairs.height}...")
        sample_id = path.basename(tab_vcf).split('.')[0]
        df_tab_vcf = pl.read_csv(tab_vcf, separator="\t") \
            .filter(
                # (pl.col("INFO/AF") < 0.001 ) &
                (pl.col("INFO/RESCUE") != ".") & 
                (pl.col("INFO/RESCUE_PROB") != ".") &
                (pl.col("INFO/RESCUE_TYPE") != ".") 
            ).with_columns(
                pl.col("INFO/vepConsequence").str.split(","),
                pl.col("INFO/vepGene").str.split(","),
                pl.col("INFO/vepSYMBOL").str.split(","),
                pl.col("INFO/vepFeature_type").str.split(","),
                pl.col("INFO/vepFeature").str.split(","),
                pl.col("INFO/vepBIOTYPE").str.split(","),
                pl.col("INFO/vepLoF").str.split(","),
                pl.col("INFO/RESCUE").str.split(","),
                pl.col("INFO/RESCUE_PROB").str.split(","),
                pl.col("INFO/RESCUE_TYPE").str.split(",")
            ).explode([
                'INFO/vepConsequence', 'INFO/vepGene', 'INFO/vepSYMBOL',
                'INFO/vepFeature_type', 'INFO/vepFeature', 'INFO/vepBIOTYPE',
                'INFO/vepLoF', 'INFO/RESCUE', 'INFO/RESCUE_PROB', 'INFO/RESCUE_TYPE'
            ]).filter(
                (pl.col("INFO/SpliceAI_Haplotype") != ".") & 
                (pl.col("INFO/vepFeature_type") == "Transcript") & 
                (pl.col("INFO/vepLoF") == "HC") & 
                (pl.col("INFO/vepBIOTYPE") == "protein_coding") &
                ((pl.col("INFO/vepConsequence") == "splice_donor_variant") | (pl.col("INFO/vepConsequence") == "splice_acceptor_variant"))
            ).with_columns(
                pl.col("INFO/SpliceAI_Haplotype").str.split(",")
            ).explode("INFO/SpliceAI_Haplotype").unique() \
            .filter((pl.col("INFO/RESCUE_PROB").str.contains(r"^\.")) | (pl.col("INFO/RESCUE_PROB").str.contains(r"\.$"))) \
            .with_columns(
                pl.col("INFO/RESCUE_PROB").str.split("&"),
                pl.col("INFO/RESCUE").str.split("&"),
                pl.col("INFO/RESCUE_TYPE").str.split("&"),
                pl.col("INFO/SpliceAI_Haplotype").str.split("&")
            ).explode(["INFO/RESCUE_PROB", "INFO/RESCUE", "INFO/SpliceAI_Haplotype", "INFO/RESCUE_TYPE"]) \
            .filter((pl.col("INFO/RESCUE_PROB") != ".") & (pl.col("INFO/RESCUE") != ".") & (pl.col("INFO/SpliceAI_Haplotype") != ".") & (pl.col("INFO/RESCUE_TYPE") != ".|.|.")) \
            .rename({
                "CHROM": "chr", "POS": "pos", "REF": "ref", "ALT": "alt", "INFO/vepFeature": "transcript_id",
                "INFO/vepConsequence": "csq", "INFO/vepGene": "gene_id", "INFO/vepSYMBOL": "symbol", 
                "INFO/RESCUE": "rescue", "INFO/RESCUE_PROB": "rescue_prob", "INFO/RESCUE_TYPE": "rescue_type", 
                "INFO/SpliceAI_Haplotype": "SpliceAI", "INFO/AF": "af"
            }).select(pl.exclude("^.*INFO.*$")) # Columns that haven't been renamed we don't want anyway

        # df_tab_vcf = df_tab_vcf.with_columns(pl.zeros(df_tab_vcf.height).alias("af"))

        df_junctions = pl \
            .read_csv(junctions, separator="\t", has_header=False, schema=junction_schema) \
            .filter(pl.col("strand") > 0) 



        for row in df_tab_vcf.rows(named=True):
            # Find exons for transcript
            # Matching on Seqid is not technically necessary but gives an easy way to disqualify most attempted matches, so should speed this up
            df_exons = df_gff.filter((pl.col("Seqid") == row["chr"]) & (pl.col("Parent") == f"transcript:{row['transcript_id']}") & (pl.col("Type") == "exon"))

            # Find exon in question
            target_exon = df_exons.filter((pl.col("Start")-2 < row['pos']) & (pl.col("End")+2 > row['pos']))

            if target_exon.shape[0] != 1: # If there is some irregularity with finding a matching exon, don't bother for now
                continue

            target_exon = target_exon.rows(named=True)[0]

            # Figure out what splice site we are dealing with
            strand = target_exon['Strand']
            rank = int(target_exon['rank'])
            max_rank = int(df_exons["rank"].max())

            splice_site = ''
            if np.abs(target_exon['Start'] - row['pos']) < 2:
                if strand == '+' and rank > 1:
                    splice_site = 'acceptor'
                elif strand == '-' and rank < max_rank:
                    splice_site = 'donor'
            elif np.abs(target_exon['End'] - row['pos']) < 2:
                if strand == '+' and rank < max_rank:
                    splice_site = 'donor'
                elif strand == '-' and rank > 1:
                    splice_site = 'acceptor'

            spliceai_fields = row['SpliceAI'].split('|')

            canonical_pos = 0
            canonical_pair_pos = 0
            competitor_pos = 0
            cryptic_pos = 0
            variant_junctions = []
            if splice_site == 'acceptor':
                if strand == '+':
                    canonical_pos = int(target_exon['Start']) - 1
                    canonical_pair_pos = int(df_exons.filter(pl.col("rank") == str(rank-1)).rows(named=True)[0]['End']) + 1

                    if rank < max_rank:
                        competitor_pos = int(df_exons.filter(pl.col("rank") == str(rank+1)).rows(named=True)[0]['Start']) - 1

                    cryptic_pos = row['pos'] + int(spliceai_fields[6]) - 1
                    if cryptic_pos == canonical_pos:
                        cryptic_pos = 0


                    # print(f"Strand = '{strand}'")
                    # print(row['chr'])
                    # print(f'Canonical  = {canonical_pair_pos}:{canonical_pos}')
                    # print(f'Competitor = {canonical_pair_pos}:{competitor_pos}')
                    # print(f'Cryptic    = {canonical_pair_pos}:{cryptic_pos}', end='\n\n')

                    variant_junctions.append(('canonical', canonical_pair_pos, canonical_pos))
                    variant_junctions.append(('competitor', canonical_pair_pos, competitor_pos))
                    variant_junctions.append(('cryptic', canonical_pair_pos, cryptic_pos))

                elif strand == '-':
                    canonical_pos = int(target_exon['End']) + 1
                    canonical_pair_pos = int(df_exons.filter(pl.col("rank") == str(rank-1)).rows(named=True)[0]['Start']) - 1

                    if rank < max_rank:
                        competitor_pos = int(df_exons.filter(pl.col("rank") == str(rank+1)).rows(named=True)[0]['End']) + 1

                    cryptic_pos = row['pos'] - int(spliceai_fields[6]) + 1
                    if cryptic_pos == canonical_pos:
                        cryptic_pos = 0



                    # print(f"Strand = '{strand}'")
                    # print(row['chr'])
                    # print(f'Canonical  = {canonical_pos}:{canonical_pair_pos}')
                    # print(f'Competitor = {competitor_pos}:{canonical_pair_pos}')
                    # print(f'Cryptic    = {cryptic_pos}:{canonical_pair_pos}', end='\n\n')

                    variant_junctions.append(('canonical', canonical_pos, canonical_pair_pos))
                    variant_junctions.append(('competitor', competitor_pos, canonical_pair_pos))
                    variant_junctions.append(('cryptic', cryptic_pos, canonical_pair_pos))

            elif splice_site == 'donor':
                if strand == '+':
                    canonical_pos = int(target_exon['End']) + 1
                    canonical_pair_pos = int(df_exons.filter(pl.col("rank") == str(rank+1)).rows(named=True)[0]['Start']) - 1

                    if rank > 1:
                        competitor_pos = int(df_exons.filter(pl.col("rank") == str(rank-1)).rows(named=True)[0]['End']) + 1

                    cryptic_pos = row['pos'] + int(spliceai_fields[8]) + 1
                    if cryptic_pos == canonical_pos:
                        cryptic_pos = 0



                    # print(f"Strand = '{strand}'")
                    # print(row['chr'])
                    # print(f'Canonical  = {canonical_pos}:{canonical_pair_pos}')
                    # print(f'Competitor = {competitor_pos}:{canonical_pair_pos}')
                    # print(f'Cryptic    = {cryptic_pos}:{canonical_pair_pos}', end='\n\n')

                    variant_junctions.append(('canonical', canonical_pos, canonical_pair_pos))
                    variant_junctions.append(('competitor', competitor_pos, canonical_pair_pos))
                    variant_junctions.append(('cryptic', cryptic_pos, canonical_pair_pos))



                elif strand == '-':
                    canonical_pos = int(target_exon['Start']) - 1
                    canonical_pair_pos = int(df_exons.filter(pl.col("rank") == str(rank+1)).rows(named=True)[0]['End']) + 1

                    if rank > 1:
                        competitor_pos = int(df_exons.filter(pl.col("rank") == str(rank-1)).rows(named=True)[0]['Start']) - 1

                    cryptic_pos = row['pos'] - int(spliceai_fields[8]) - 1
                    if cryptic_pos == canonical_pos:
                        cryptic_pos = 0


                    # print(f"Strand = '{strand}'")
                    # print(row['chr'])
                    # print(f'Canonical  = {canonical_pair_pos}:{canonical_pos}')
                    # print(f'Competitor = {canonical_pair_pos}:{competitor_pos}')
                    # print(f'Cryptic    = {canonical_pair_pos}:{cryptic_pos}', end='\n\n')

                    variant_junctions.append(('canonical', canonical_pair_pos, canonical_pos))
                    variant_junctions.append(('competitor', canonical_pair_pos, competitor_pos))
                    variant_junctions.append(('cryptic', canonical_pair_pos, cryptic_pos))


            for j in variant_junctions:
                if j[1] == 0 or j[2] == 0:
                    output.append([row['chr'], row['pos'], row['ref'], row['alt'], row['transcript_id'], row['gene_id'], row['symbol'], sample_id, splice_site, row['af'], j[0], row['rescue'], row['rescue_prob'], row['rescue_type'], row['SpliceAI'], np.NaN])
                    continue

                support = df_junctions.filter((pl.col("chr") == row['chr'][3:]) & (pl.col("start") == j[1]) & (pl.col("end") == j[2]))

                if (support.shape[0] != 1):
                    output.append([row['chr'], row['pos'], row['ref'], row['alt'], row['transcript_id'], row['gene_id'], row['symbol'], sample_id, splice_site, row['af'], j[0], row['rescue'], row['rescue_prob'], row['rescue_type'], row['SpliceAI'], 0])
                    continue


                output.append([
                    row['chr'], row['pos'], row['ref'], row['alt'], row['transcript_id'], row['gene_id'], row['symbol'], sample_id, splice_site, row['af'], j[0], row['rescue'], row['rescue_prob'], row['rescue_type'], row['SpliceAI'], support.rows(named=True)[0]['unique_junction_reads']
                ])

    df_output = pl.DataFrame(np.array(output), schema={'chr': pl.Utf8, 'pos': pl.Int64, 'ref': pl.Utf8, 'alt': pl.Utf8, 'transcript_id': pl.Utf8, 'gene_id': pl.Utf8, 'symbol': pl.Utf8, 'sample': pl.Utf8, 'splice_site_type': pl.Utf8, 'AF': pl.Float32, 'junction_type': pl.Utf8, 'rescue': pl.Int8, 'rescue_prob': pl.Float32, 'rescue_type': pl.Utf8, 'spliceai_hap': pl.Utf8, 'n_reads': pl.Float64}) \
        .sort(["chr", "pos", "ref", "alt", "transcript_id", "sample", "junction_type"])

    print(df_output)

    df_output.write_csv("junction_table.tsv", separator="\t")


if __name__ == "__main__":
    main()

