// What is being dragged right now (a chip, or a model's letter badge), shared
// by the model cards and the machine view so boxes can say whether a drop works.

import type { Subject } from './pending'

export interface Dragging {
  subject: Subject
  model: string
  /** the subsystem id of a chip */
  sub: string
  /** the box a chip comes from; null for a whole model */
  fromBox: string | null
}

export const dnd = $state<{ active: Dragging | null }>({ active: null })
