import { Box, Heading, Text } from '@chakra-ui/react';
import { SECTION_META, type ChartSection } from './registry';

interface SectionHeadingProps {
  section: ChartSection;
}

/**
 * Heading + subtitle for the per-section subgroups on the Analysis tab.
 * Lives in its own file so registry.tsx (which is data + helpers) doesn't
 * mix component and non-component exports — that pattern breaks Vite's
 * fast-refresh.
 */
export function SectionHeading({ section }: SectionHeadingProps) {
  const meta = SECTION_META[section];
  return (
    <Box mb={2} mt={2}>
      <Heading size="sm" color="gray.700">
        {meta.title}
      </Heading>
      <Text fontSize="xs" color="gray.500">
        {meta.subtitle}
      </Text>
    </Box>
  );
}
