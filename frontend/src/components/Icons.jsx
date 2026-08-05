const baseProps = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

export function RiceMark({ size = 38 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 42 42" fill="none" aria-hidden="true">
      <path d="M20.5 38V8.5M20.5 13C14 10.5 10 6 8 2M20.5 19C28 15 31 10 32 5M20.5 24.5C13 22 8.5 17.5 6 13M20.5 31C29 28.5 34 23 36 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M10.5 6.2c4.5.8 7.7 3 9.7 6.2-4.2-.2-7.3-2.3-9.7-6.2ZM30.4 8.6c-1.2 3.8-4.2 7-8.6 8.9.8-4.2 3.6-7.2 8.6-8.9ZM8.7 16.4c4.6.6 8.2 2.7 10.8 6.2-4.6.1-8.2-2-10.8-6.2ZM33.2 20.3c-1.1 4.4-4.3 7.5-10 9.5 1.5-4.8 4.8-8 10-9.5Z" fill="currentColor" opacity=".88" />
    </svg>
  );
}

export function PlusIcon() {
  return <svg {...baseProps}><path d="M12 5v14M5 12h14" /></svg>;
}
export function ChatIcon() {
  return <svg {...baseProps}><path d="M20 15a4 4 0 0 1-4 4H8l-4 3V7a4 4 0 0 1 4-4h8a4 4 0 0 1 4 4v8Z" /><path d="M8 10h.01M12 10h.01M16 10h.01" /></svg>;
}
export function BookIcon() {
  return <svg {...baseProps}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V4H6.5A2.5 2.5 0 0 0 4 6.5v13Z" /><path d="M8 7h8M8 11h6" /></svg>;
}
export function ChartIcon() {
  return <svg {...baseProps}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>;
}
export function ImageIcon() {
  return <svg {...baseProps}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="2" /><path d="m21 15-4.5-4.5L7 20" /></svg>;
}
export function SendIcon() {
  return <svg {...baseProps}><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></svg>;
}
export function BotIcon() {
  return <svg {...baseProps}><rect x="4" y="7" width="16" height="12" rx="4" /><path d="M12 3v4M8.5 12h.01M15.5 12h.01M9 16h6" /></svg>;
}
export function CheckIcon() {
  return <svg {...baseProps}><path d="m5 12 4 4L19 6" /></svg>;
}
export function CloseIcon() {
  return <svg {...baseProps}><path d="m6 6 12 12M18 6 6 18" /></svg>;
}
export function ThumbsUpIcon() {
  return <svg {...baseProps}><path d="M7 10v10H3V10h4ZM7 18c2 0 3.5 2 7 2h2.5a2 2 0 0 0 2-1.6l1.2-6A2 2 0 0 0 17.8 10H14l.6-3A3.4 3.4 0 0 0 13 3l-6 7" /></svg>;
}
export function ThumbsDownIcon() {
  return <svg {...baseProps}><path d="M7 14V4H3v10h4ZM7 6c2 0 3.5-2 7-2h2.5a2 2 0 0 1 2 1.6l1.2 6a2 2 0 0 1-1.9 2.4H14l.6 3A3.4 3.4 0 0 1 13 21l-6-7" /></svg>;
}
export function WarningIcon() {
  return <svg {...baseProps}><path d="M12 3 2.5 20h19L12 3Z" /><path d="M12 9v4M12 17h.01" /></svg>;
}
export function MenuIcon() {
  return <svg {...baseProps}><path d="M4 7h16M4 12h16M4 17h16" /></svg>;
}
