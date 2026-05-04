/* hmmseqdb: create a protein dsqdata database for GPU hmmsearch.
 */
#include <p7_config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_dsqdata.h"
#include "esl_getopts.h"
#include "esl_sqio.h"

#include "hmmer.h"

static ESL_OPTIONS options[] = {
  /* name           type       default env  range toggles reqs incomp help                                            docgroup */
  { "-h",           eslARG_NONE, FALSE, NULL, NULL, NULL,   NULL, NULL, "show brief help on version and usage",        1 },
  { "--informat",   eslARG_STRING, NULL, NULL, NULL, NULL,  NULL, NULL, "assert input <seqfile> is in format <s>",     2 },
  { "--check",      eslARG_NONE, FALSE, NULL, NULL, NULL,   NULL, NULL, "validate the written dsqdata database",       2 },
  {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static char usage[]  = "[options] <seqfile> <seqdb>";
static char banner[] = "build a protein sequence database for hmmsearch --gpu";

static int
process_commandline(int argc, char **argv, ESL_GETOPTS **ret_go, char **ret_seqfile, char **ret_seqdb)
{
  ESL_GETOPTS *go = esl_getopts_Create(options);
  int          status;

  if (esl_opt_ProcessEnvironment(go)         != eslOK)  { if (printf("Failed to process environment: %s\n", go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if (esl_opt_ProcessCmdline(go, argc, argv) != eslOK)  { if (printf("Failed to parse command line: %s\n",  go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if (esl_opt_VerifyConfig(go)               != eslOK)  { if (printf("Failed to parse command line: %s\n",  go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  if (esl_opt_GetBoolean(go, "-h") == TRUE)
    {
      p7_banner(stdout, argv[0], banner);
      esl_usage(stdout, argv[0], usage);
      if (puts("\nBasic options:") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 1, 2, 80);
      if (puts("\nOther options:") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
      esl_opt_DisplayHelp(stdout, go, 2, 2, 80);
      exit(0);
    }

  if (esl_opt_ArgNumber(go)                  != 2)     { if (puts("Incorrect number of command line arguments.")      < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if ((*ret_seqfile = esl_opt_GetArg(go, 1)) == NULL)  { if (puts("Failed to get <seqfile> argument on command line") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if ((*ret_seqdb   = esl_opt_GetArg(go, 2)) == NULL)  { if (puts("Failed to get <seqdb> argument on command line")   < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  if (strcmp(*ret_seqfile, "-") == 0)
    { if (puts("<seqfile> must be a rewindable file; '-' is not supported") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  *ret_go = go;
  return eslOK;

 FAILURE:
  esl_usage(stdout, argv[0], usage);
  if (puts("\nwhere basic options are:") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
  esl_opt_DisplayHelp(stdout, go, 1, 2, 80);
  if (printf("\nTo see more help on available options, do %s -h\n\n", argv[0]) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
  esl_getopts_Destroy(go);
  exit(1);

 ERROR:
  if (go) esl_getopts_Destroy(go);
  exit(status);
}

int
main(int argc, char **argv)
{
  ESL_GETOPTS  *go      = NULL;
  ESL_ALPHABET *abc     = NULL;
  ESL_SQFILE   *sqfp    = NULL;
  ESL_DSQDATA  *dd      = NULL;
  ESL_DSQDATA_CHUNK *chu = NULL;
  char         *seqfile = NULL;
  char         *seqdb   = NULL;
  char          errbuf[eslERRBUFSIZE];
  int           fmt     = eslSQFILE_UNKNOWN;
  int           status;
  uint64_t      check_nseq = 0;
  uint64_t      check_nres = 0;
  int           i;

  process_commandline(argc, argv, &go, &seqfile, &seqdb);

  if (esl_opt_IsOn(go, "--informat")) {
    fmt = esl_sqio_EncodeFormat(esl_opt_GetString(go, "--informat"));
    if (fmt == eslSQFILE_UNKNOWN) p7_Fail("%s is not a recognized input sequence file format\n", esl_opt_GetString(go, "--informat"));
  }

  abc = esl_alphabet_Create(eslAMINO);
  if (abc == NULL) p7_Fail("Failed to create protein alphabet\n");

  status = esl_sqfile_OpenDigital(abc, seqfile, fmt, p7_SEQDBENV, &sqfp);
  if      (status == eslENOTFOUND) p7_Fail("Failed to open sequence file %s for reading\n",          seqfile);
  else if (status == eslEFORMAT)   p7_Fail("Sequence file %s is empty or misformatted\n",            seqfile);
  else if (status == eslEINVAL)    p7_Fail("Can't autodetect format of a stdin or .gz seqfile");
  else if (status != eslOK)        p7_Fail("Unexpected error %d opening sequence file %s\n", status, seqfile);

  printf("Working...    ");
  fflush(stdout);

  status = esl_dsqdata_Write(sqfp, seqdb, errbuf);
  if      (status == eslEWRITE)         p7_Fail("Failed to write dsqdata files:\n  %s\n", errbuf);
  else if (status == eslEFORMAT)        p7_Fail("Parse failed (sequence file %s):\n%s\n", seqfile, errbuf);
  else if (status == eslEUNIMPLEMENTED) p7_Fail("Failed to create GPU sequence database: unsupported large protein sequence\n");
  else if (status != eslOK)             p7_Fail("Unexpected error %d while creating GPU sequence database\n", status);

  printf("done.\n");
  printf("Created protein dsqdata database %s for hmmsearch --gpu.\n", seqdb);

  if (esl_opt_GetBoolean(go, "--check")) {
    status = esl_dsqdata_Open(&abc, seqdb, 1, &dd);
    if      (status == eslENOTFOUND) p7_Fail("Failed to open created dsqdata database %s for checking: %s\n", seqdb, dd ? dd->errbuf : "");
    else if (status == eslEFORMAT)   p7_Fail("Created dsqdata database %s failed format check: %s\n", seqdb, dd ? dd->errbuf : "");
    else if (status != eslOK)        p7_Fail("Unexpected error %d opening created dsqdata database %s\n", status, seqdb);

    while ((status = esl_dsqdata_Read(dd, &chu)) == eslOK) {
      check_nseq += chu->N;
      for (i = 0; i < chu->N; i++) check_nres += chu->L[i];
      esl_dsqdata_Recycle(dd, chu);
      chu = NULL;
    }
    if (status != eslEOF) p7_Fail("Read/check failed for created dsqdata database %s: %s\n", seqdb, dd->errbuf);
    if (check_nseq != dd->nseq || check_nres != dd->nres)
      p7_Fail("Created dsqdata database %s failed count check: header has %" PRIu64 " seqs/%" PRIu64 " residues, read %" PRIu64 " seqs/%" PRIu64 " residues\n",
              seqdb, dd->nseq, dd->nres, check_nseq, check_nres);
    printf("Checked protein dsqdata database %s: %" PRIu64 " sequences, %" PRIu64 " residues, max length %" PRIu64 ".\n",
           seqdb, dd->nseq, dd->nres, dd->max_seqlen);
    esl_dsqdata_Close(dd);
    dd = NULL;
  }

  esl_sqfile_Close(sqfp);
  esl_alphabet_Destroy(abc);
  esl_getopts_Destroy(go);
  return eslOK;
}
