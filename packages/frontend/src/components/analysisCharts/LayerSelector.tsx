import {
  ButtonGroup,
  Button,
  HStack,
  Text,
  Tooltip,
  Box,
} from '@chakra-ui/react';
import { Layers } from 'lucide-react';
import { LAYER_OPTIONS } from './layerOptions';

interface LayerSelectorProps {
  value: string;
  onChange: (next: string) => void;
}

/**
 * Compact, sticky-friendly layer toggle. Charts that respect `layerAware`
 * read the value from the parent page and re-render; others ignore it.
 */
export function LayerSelector({ value, onChange }: LayerSelectorProps) {
  return (
    <HStack spacing={2}>
      <HStack spacing={1} color="gray.600">
        <Box as={Layers} boxSize={4} />
        <Text fontSize="xs" fontWeight="medium">
          Layer
        </Text>
      </HStack>
      <ButtonGroup size="xs" isAttached variant="outline">
        {LAYER_OPTIONS.map((opt) => (
          <Tooltip key={opt.value} label={opt.hint} placement="bottom" hasArrow>
            <Button
              colorScheme={value === opt.value ? 'blue' : 'gray'}
              variant={value === opt.value ? 'solid' : 'outline'}
              onClick={() => onChange(opt.value)}
            >
              {opt.label}
            </Button>
          </Tooltip>
        ))}
      </ButtonGroup>
    </HStack>
  );
}
