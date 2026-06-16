// Curated sample images bundled with the app under client/public/samples/.
// Each entry carries optional suggested prompts per modality so the playground
// pickers can pre-fill a reasonable query. Replace these with industry- or
// customer-specific samples by dropping new files into client/public/samples/
// and updating this manifest.

export interface SampleImage {
  /** Public URL — must match a file in client/public/samples/. */
  src: string;
  /** Display name in the picker. */
  name: string;
  /** Short caption shown under the thumbnail. */
  caption: string;
  /** Optional prompts the picker can pre-fill on the destination playground. */
  prompts?: {
    segmentation?: string;
    detection?: string;
    embedding?: string;
  };
}

export const SAMPLES: SampleImage[] = [
  {
    src: '/samples/warehouse-pallets.jpg',
    name: 'Warehouse pallets',
    caption: 'Pallets stacked in a warehouse',
    prompts: {
      segmentation: 'pallet',
      detection: 'pallet. box. forklift.',
      embedding: 'pallets stacked in a warehouse',
    },
  },
  {
    src: '/samples/grocery-shelf.jpg',
    name: 'Grocery shelf',
    caption: 'CPG products on a retail shelf',
    prompts: {
      segmentation: 'bottle',
      detection: 'bottle. box. package.',
      embedding: 'CPG products on a supermarket shelf',
    },
  },
  {
    src: '/samples/factory-worker.jpg',
    name: 'Factory worker',
    caption: 'Worker in PPE on a production line',
    prompts: {
      segmentation: 'person',
      detection: 'person. helmet. uniform.',
      embedding: 'worker in safety gear on a factory line',
    },
  },
  {
    src: '/samples/kitchen-counter.jpg',
    name: 'Kitchen counter',
    caption: 'Everyday objects on a kitchen counter',
    prompts: {
      segmentation: 'bottle',
      detection: 'bottle. cup. plate.',
      embedding: 'kitchen counter with cookware',
    },
  },
  {
    src: '/samples/highway-cars.jpg',
    name: 'Highway',
    caption: 'Cars on a highway at dusk',
    prompts: {
      segmentation: 'car',
      detection: 'car. truck. sign.',
      embedding: 'cars on a highway',
    },
  },
  {
    src: '/samples/office-desk.jpg',
    name: 'Office desk',
    caption: 'Laptop, keyboard, plant on a desk',
    prompts: {
      segmentation: 'laptop',
      detection: 'laptop. keyboard. cup.',
      embedding: 'modern home office desk',
    },
  },
];
