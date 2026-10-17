import { create } from "zustand";

type Theme = "dark" | "light";
type UpDown = "red" | "green";

interface UIState {
  theme: Theme;
  upDown: UpDown;
  aiOpen: boolean;
  selectedCode: string;
  setTheme: (t: Theme) => void;
  setUpDown: (u: UpDown) => void;
  toggleAi: (open?: boolean) => void;
  select: (code: string) => void;
}

function load<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T) || fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, v: string) {
  try {
    localStorage.setItem(key, v);
  } catch {
    /* storage may be unavailable (private mode) */
  }
}

export const useUI = create<UIState>((set) => ({
  theme: load<Theme>("defaco.theme", "dark"),
  upDown: load<UpDown>("defaco.updown", "red"),
  aiOpen: false,
  selectedCode: "600519.SH",
  setTheme: (theme) => {
    save("defaco.theme", theme);
    document.documentElement.dataset.theme = theme;
    set({ theme });
  },
  setUpDown: (upDown) => {
    save("defaco.updown", upDown);
    document.documentElement.dataset.updown = upDown;
    set({ upDown });
  },
  toggleAi: (open) => set((s) => ({ aiOpen: open ?? !s.aiOpen })),
  select: (selectedCode) => set({ selectedCode }),
}));

/** Read a CSS custom property (charts need concrete colours). */
export const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
