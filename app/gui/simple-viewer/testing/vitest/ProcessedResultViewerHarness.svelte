<script lang="ts">
  import ProcessedResultViewer from "../../src/components/ProcessedResultViewer.svelte";

  let didClose = false;

  const result = {
    inputPath: "/input/sub01.nii.gz",
    outputPath: "/output/sub01.mgz",
    outputFilename: "sub01.mgz",
    runResult: "ok"
  };
</script>

<ProcessedResultViewer
  {result}
  on:close={() => {
    didClose = true;
  }}
/>

{#if didClose}
  <span data-testid="did-close">closed</span>
{/if}
