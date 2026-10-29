import nodeLogger from 'astro/logger/node';

// Astro reports empty configured collections as missing, but they are valid here.
const emptyCollectionWarnings = new Set(['reports', 'notes'].map(collection =>
  `The collection ${JSON.stringify(collection)} does not exist or is empty. Please check your content config file for errors.`,
));

export default function createLogger(options) {
  const destination = nodeLogger(options);
  return {
    write(event) {
      if (event.level === 'warn' && event.label === 'content' && emptyCollectionWarnings.has(event.message)) return;
      destination.write(event);
    },
  };
}
