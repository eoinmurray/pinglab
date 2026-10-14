import { useEffect, useState } from 'react';
import { defaultFilter } from 'cmdk';
import { CommandDialog, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from './ui/command';
import type { NavigationEntry } from '../lib/navigation';
import '../styles/command.css';

interface Props {
  collections: NavigationEntry[];
  articles: NavigationEntry[];
}

function navigationFilter(_value: string, search: string, keywords: string[] = []) {
  const query = search.trim().toLowerCase();
  if (!query) return 1;
  const [title = '', id = '', ...metadata] = keywords.map(keyword => keyword.toLowerCase());
  const experimentNumber = /^exp(\d+)$/.exec(id)?.[1];
  if (/^\d+$/.test(query) && experimentNumber && Number(query) === Number(experimentNumber)) return .99;
  if (title === query) return 1;
  if (title.startsWith(query)) return .95;
  if (title.includes(query)) return .85;
  if (id === query) return .9;
  if (id.startsWith(query)) return .8;
  const titleScore = defaultFilter(title, query);
  if (titleScore >= .2) return titleScore * .6;
  const description = metadata.join(' ');
  return query.split(/\s+/).every(word => description.includes(word)) ? .2 : 0;
}

export default function CommandPalette({ collections, articles }: Props) {
  const [open, setOpen] = useState(false);
  const [shortcut, setShortcut] = useState('Ctrl K');
  useEffect(() => {
    setShortcut(/Mac|iPhone|iPad/.test(navigator.platform) ? '⌘ K' : 'Ctrl K');
    function onKeyDown(event: KeyboardEvent) {
      if (event.key.toLowerCase() === 'k' && (event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!event.repeat) setOpen(value => !value);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  function items(entries: NavigationEntry[]) {
    return entries.map(entry => (
      <CommandItem key={entry.href} value={entry.href} keywords={[entry.title, ...entry.keywords]}
        onSelect={() => { setOpen(false); window.location.assign(entry.href); }}>
        <span className="command-item-title">{entry.title}</span>
        <span className="command-item-detail">{entry.detail}</span>
      </CommandItem>
    ));
  }

  return (
    <CommandDialog open={open} onOpenChange={setOpen} filter={navigationFilter} trigger={
        <button type="button" className="command-trigger" aria-label={`Search articles and collections (${shortcut})`}>
          <span className="command-shortcut-icon" aria-hidden="true">⌘K</span>
          <span>Search</span>
        </button>
      }>
      <CommandInput autoFocus placeholder="Search articles and collections…" aria-label="Search articles and collections" />
      <CommandList label="Matching articles and collections">
        <CommandEmpty>No articles or collections found.</CommandEmpty>
        <CommandGroup heading="Navigation">
          {items([{ href: '/', title: 'Home', detail: 'Homepage', keywords: ['home', 'homepage', 'Pinglab'] }])}
        </CommandGroup>
        <CommandGroup heading="Collections">{items(collections)}</CommandGroup>
        <CommandGroup heading="Articles">{items(articles)}</CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
