import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverHeader,
  PopoverBody,
  PopoverFooter,
  Button,
  VStack,
  Checkbox,
  Text,
  Box,
  HStack,
  Badge,
} from '@chakra-ui/react';
import { Settings } from 'lucide-react';
import { CHART_REGISTRY, type ChartDescriptor, type ChartTab } from './registry';

interface ChartPickerProps {
  hiddenIds: string[];
  onToggle: (id: string) => void;
  onReset: () => void;
}

const TAB_LABELS: Record<ChartTab, string> = {
  diagnostics: 'Diagnostics',
  statistics: 'Statistics',
  analysis: 'Analysis',
};

export function ChartPicker({ hiddenIds, onToggle, onReset }: ChartPickerProps) {
  const hiddenSet = new Set(hiddenIds);

  // Group registry entries by tab, preserving in-tab ordering
  const grouped = CHART_REGISTRY.reduce<Partial<Record<ChartTab, ChartDescriptor[]>>>(
    (acc, c) => {
      if (!acc[c.tab]) acc[c.tab] = [];
      acc[c.tab]!.push(c);
      return acc;
    },
    {},
  );

  const nHidden = hiddenSet.size;

  return (
    <Popover placement="bottom-end">
      <PopoverTrigger>
        <Button size="sm" leftIcon={<Settings size={14} />} variant="outline">
          Customize
          {nHidden > 0 && (
            <Badge ml={2} colorScheme="orange" fontSize="2xs">
              {nHidden} hidden
            </Badge>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent w="340px">
        <PopoverHeader fontWeight="bold" fontSize="sm">
          Analysis Charts
        </PopoverHeader>
        <PopoverBody maxH="460px" overflowY="auto">
          <VStack align="stretch" spacing={4}>
            {(Object.keys(grouped) as ChartTab[]).map((tab) => {
              const charts = grouped[tab] ?? [];
              if (charts.length === 0) return null;
              return (
              <Box key={tab}>
                <HStack mb={2}>
                  <Text
                    fontSize="xs"
                    fontWeight="bold"
                    color="gray.500"
                    textTransform="uppercase"
                    letterSpacing="wide"
                  >
                    {TAB_LABELS[tab] ?? tab}
                  </Text>
                  <Text fontSize="xs" color="gray.400">
                    ({charts.length})
                  </Text>
                </HStack>
                <VStack align="stretch" spacing={1} pl={1}>
                  {charts.map((c) => (
                    <Checkbox
                      key={c.id}
                      size="sm"
                      isChecked={!hiddenSet.has(c.id)}
                      onChange={() => onToggle(c.id)}
                    >
                      <VStack align="start" spacing={0}>
                        <Text fontSize="sm">{c.title}</Text>
                        {c.description && (
                          <Text fontSize="xs" color="gray.500">
                            {c.description}
                          </Text>
                        )}
                      </VStack>
                    </Checkbox>
                  ))}
                </VStack>
              </Box>
              );
            })}
          </VStack>
        </PopoverBody>
        <PopoverFooter>
          <Button size="xs" variant="ghost" onClick={onReset} w="full" isDisabled={nHidden === 0}>
            Show All
          </Button>
        </PopoverFooter>
      </PopoverContent>
    </Popover>
  );
}
