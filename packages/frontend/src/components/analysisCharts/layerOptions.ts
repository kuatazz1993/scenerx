export interface LayerOption {
  value: string;
  label: string;
  hint: string;
}

export const LAYER_OPTIONS: LayerOption[] = [
  { value: 'full', label: 'Full', hint: 'Whole-image values (no FMB split)' },
  { value: 'foreground', label: 'FG', hint: 'Foreground layer (within ~5 m)' },
  { value: 'middleground', label: 'MG', hint: 'Middleground layer' },
  { value: 'background', label: 'BG', hint: 'Background layer' },
];
