<script lang="ts">
  export let searchPlaceholder: string;
  export let onHome: () => void;

  const controls = ["notifications", "settings", "help"];
</script>

<header class="fixed left-0 top-0 z-50 flex w-full items-center justify-between border-b border-outline-variant/15 bg-background px-4 py-3 md:px-6">
  <div class="flex items-center gap-4 md:gap-8">
    <button class="font-headline text-xl font-bold tracking-tight text-tertiary" type="button" on:click={onHome}>
      Damadian Imaging
    </button>
    <div class="hidden items-center gap-3 rounded-lg bg-surface-container-highest px-3 py-1.5 md:flex">
      <span aria-hidden="true" class="material-symbols-outlined text-sm text-primary">search</span>
      <input
        aria-label="Search workspace"
        class="w-56 border-none bg-transparent text-sm text-on-surface placeholder:text-on-surface-variant/50 focus:outline-none lg:w-64"
        placeholder={searchPlaceholder}
        type="text"
      />
    </div>
  </div>

  <div class="flex items-center gap-2 md:gap-4">
    {#each controls as controlIcon}
      <button aria-label={controlIcon} class="rounded-full p-2 text-primary/70 transition-colors duration-200 hover:bg-surface-container-high hover:text-primary" type="button">
        <span aria-hidden="true" class="material-symbols-outlined">{controlIcon}</span>
      </button>
    {/each}
    <div class="h-8 w-8 overflow-hidden rounded-full border border-tertiary/20 bg-primary-container">
      <img
        alt="Clinical researcher avatar"
        class="h-full w-full object-cover"
        src="https://lh3.googleusercontent.com/aida-public/AB6AXuAkbRSWsTYl5aziF0Cx7aHhNdCgE3ff5sw_Glg6pF_A2qxG46KpbEDSkLsxGWr5vJIpZX3idtO_-sVkMubojlV7RVoKdaXpRvqxdQXHnphHLBgY6loQD03PhmigzkV_lA37DMYBOZLkCtpKT0Y11_6MLsbjJrr9PCyOrhBsgPxsNWn9VIsJhTIRtSIONXnO1Caeq05jxRKSZqMeJdtdasfShqL2KskTibRt74rebgnhnj28gs7MGjBdAij7HuFmuhjOS6KmeaiFp4Kj"
      />
    </div>
  </div>
</header>
