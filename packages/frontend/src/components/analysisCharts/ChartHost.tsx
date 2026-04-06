import {
  Card,
  CardHeader,
  CardBody,
  Heading,
  HStack,
  IconButton,
  Tooltip,
} from '@chakra-ui/react';
import { X } from 'lucide-react';
import type { ChartDescriptor } from './registry';
import type { ChartContext } from './ChartContext';

interface ChartHostProps {
  descriptor: ChartDescriptor;
  ctx: ChartContext;
  onHide: (id: string) => void;
}

/**
 * Wraps a single ChartDescriptor in a Chakra Card. Returns null when the
 * descriptor's data isn't available, so callers can just `.map()` over the
 * full registry without guards.
 */
export function ChartHost({ descriptor, ctx, onHide }: ChartHostProps) {
  if (!descriptor.isAvailable(ctx)) return null;
  return (
    <Card>
      <CardHeader pb={2}>
        <HStack justify="space-between" align="start">
          <Heading size="sm">{descriptor.title}</Heading>
          <Tooltip label="Hide this chart" placement="left">
            <IconButton
              aria-label={`Hide ${descriptor.title}`}
              icon={<X size={14} />}
              size="xs"
              variant="ghost"
              onClick={() => onHide(descriptor.id)}
            />
          </Tooltip>
        </HStack>
      </CardHeader>
      <CardBody pt={2}>{descriptor.render(ctx)}</CardBody>
    </Card>
  );
}
