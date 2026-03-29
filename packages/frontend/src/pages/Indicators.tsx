import { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Heading,
  Button,
  VStack,
  HStack,
  Box,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Text,
  Badge,
  Checkbox,
  CheckboxGroup,
  Alert,
  AlertIcon,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Tag,
  TagLabel,
  Wrap,
  WrapItem,
  Progress,
  Divider,
  List,
  ListItem,
  Collapse,
  useDisclosure,
  Spinner,
} from '@chakra-ui/react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Lightbulb } from 'lucide-react';
import { useKnowledgeBaseSummary, useProject } from '../hooks/useApi';
import api from '../api';
import type { IndicatorRecommendation, RecommendationResponse } from '../types';
import useAppStore from '../store/useAppStore';
import useAppToast from '../hooks/useAppToast';
import PageShell from '../components/PageShell';
import PageHeader from '../components/PageHeader';
import EmptyState from '../components/EmptyState';

// Performance dimensions
const DIMENSIONS = [
  { id: 'PRF_AES', name: 'Aesthetics & Landscape Preference' },
  { id: 'PRF_RST', name: 'Stress Relief & Restoration' },
  { id: 'PRF_EMO', name: 'Emotion & Cognition' },
  { id: 'PRF_THR', name: 'Thermal Comfort' },
  { id: 'PRF_USE', name: 'Spatial Use & Activity' },
  { id: 'PRF_SOC', name: 'Social Interaction' },
];

// ── Memoized recommendation row ──────────────────────────────────────────
const RecommendationItem = memo(function RecommendationItem({
  rec,
  selected,
  onToggle,
}: {
  rec: IndicatorRecommendation;
  selected: boolean;
  onToggle: (rec: IndicatorRecommendation) => void;
}) {
  const handleCheck = useCallback(
    (e: React.MouseEvent) => { e.stopPropagation(); onToggle(rec); },
    [rec, onToggle],
  );

  return (
    <AccordionItem>
      <AccordionButton>
        <HStack flex="1" justify="space-between" pr={2}>
          <HStack>
            <Checkbox
              isChecked={selected}
              onChange={() => onToggle(rec)}
              onClick={handleCheck}
            />
            <Badge colorScheme="blue">{rec.indicator_id}</Badge>
            <Text fontWeight="bold" noOfLines={1}>{rec.indicator_name}</Text>
          </HStack>
          <HStack>
            <Progress value={rec.relevance_score * 100} size="sm" w="60px" colorScheme="green" />
            <Text fontSize="sm">{(rec.relevance_score * 100).toFixed(0)}%</Text>
          </HStack>
        </HStack>
        <AccordionIcon />
      </AccordionButton>
      <AccordionPanel pb={4}>
        <VStack align="stretch" spacing={3}>
          <Text fontSize="sm">{rec.rationale}</Text>
          <HStack flexWrap="wrap">
            {rec.rank > 0 && <Badge colorScheme="purple">#{rec.rank}</Badge>}
            <Badge colorScheme={rec.relationship_direction === 'positive' || rec.relationship_direction === 'INCREASE' ? 'green' : 'orange'}>
              {rec.relationship_direction}
            </Badge>
            {rec.strength_score && (
              <Badge colorScheme={rec.strength_score === 'A' ? 'green' : rec.strength_score === 'B' ? 'blue' : 'gray'}>
                Strength {rec.strength_score}
              </Badge>
            )}
            <Badge colorScheme={rec.confidence === 'high' ? 'green' : 'yellow'}>
              {rec.confidence} confidence
            </Badge>
            {rec.transferability_summary && (
              <Badge colorScheme="teal" variant="outline">
                Transfer: {rec.transferability_summary.high_count}H/{rec.transferability_summary.moderate_count}M/{rec.transferability_summary.low_count}L
              </Badge>
            )}
          </HStack>

          {/* Evidence citations */}
          {rec.evidence_citations && rec.evidence_citations.length > 0 ? (
            <Box>
              <Text fontSize="xs" fontWeight="bold" color="gray.600" mb={1}>Evidence Citations:</Text>
              <VStack align="stretch" spacing={1}>
                {rec.evidence_citations.map((cit) => (
                  <HStack key={cit.evidence_id} fontSize="xs" color="gray.600" align="start" spacing={2}>
                    <Badge size="sm" variant="outline" colorScheme="gray" flexShrink={0}>
                      {cit.evidence_id}
                    </Badge>
                    <Text noOfLines={2} flex={1}>
                      {cit.citation || 'No citation text'}
                      {cit.year ? ` (${cit.year})` : ''}
                    </Text>
                    {cit.direction && (
                      <Badge size="sm" colorScheme={cit.direction === 'positive' ? 'green' : 'orange'} flexShrink={0}>
                        {cit.direction}
                      </Badge>
                    )}
                  </HStack>
                ))}
              </VStack>
            </Box>
          ) : rec.evidence_ids.length > 0 ? (
            <Text fontSize="xs" color="gray.500">
              Evidence: {rec.evidence_ids.slice(0, 3).join(', ')}
              {rec.evidence_ids.length > 3 && ` +${rec.evidence_ids.length - 3} more`}
            </Text>
          ) : null}
        </VStack>
      </AccordionPanel>
    </AccordionItem>
  );
});

function Indicators() {
  const { projectId: routeProjectId } = useParams<{ projectId: string }>();
  const { data: kbSummary, isLoading: kbLoading } = useKnowledgeBaseSummary();
  const { data: routeProject } = useProject(routeProjectId || '');
  const toast = useAppToast();

  const { currentProject, selectedIndicators, addSelectedIndicator, removeSelectedIndicator, clearSelectedIndicators, recommendations, setRecommendations, indicatorRelationships, setIndicatorRelationships, recommendationSummary, setRecommendationSummary } = useAppStore();

  const activeProject = routeProject || currentProject;

  // Streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const [streamingText, setStreamingText] = useState('');
  const streamBoxRef = useRef<HTMLDivElement>(null);

  // O(1) selection lookup
  const selectedIds = useMemo(
    () => new Set(selectedIndicators.map((i) => i.indicator_id)),
    [selectedIndicators],
  );

  // Advanced options toggle
  const { isOpen: advancedOpen, onToggle: toggleAdvanced } = useDisclosure();

  // Form state — dimensions can be adjusted, rest comes from project
  const [selectedDimensions, setSelectedDimensions] = useState<string[]>([]);

  // Pre-fill dimensions from project (only when project ID changes)
  const activeProjectId = activeProject?.id;
  useEffect(() => {
    if (activeProject?.performance_dimensions) {
      setSelectedDimensions(activeProject.performance_dimensions);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProjectId]);

  // Auto-scroll streaming box
  useEffect(() => {
    if (streamBoxRef.current) {
      streamBoxRef.current.scrollTop = streamBoxRef.current.scrollHeight;
    }
  }, [streamingText]);

  // recommendations now from store (persisted across navigation)

  const handleRecommend = async () => {
    const name = activeProject?.project_name || '';
    if (!name || selectedDimensions.length === 0) {
      toast({ title: 'Please select performance dimensions', status: 'warning' });
      return;
    }

    setIsStreaming(true);
    setStreamingText('');
    setStreamStatus('Starting…');

    try {
      await api.indicators.recommendStream(
        {
          project_name: name,
          project_location: activeProject?.project_location || '',
          space_type_id: activeProject?.space_type_id || '',
          koppen_zone_id: activeProject?.koppen_zone_id || '',
          lcz_type_id: activeProject?.lcz_type_id || '',
          age_group_id: activeProject?.age_group_id || '',
          performance_dimensions: selectedDimensions,
          subdimensions: activeProject?.subdimensions || [],
          design_brief: activeProject?.design_brief || '',
        },
        (event) => {
          if (event.type === 'status') {
            setStreamStatus(event.message || '');
          } else if (event.type === 'chunk') {
            setStreamingText((prev) => prev + (event.text || ''));
          } else if (event.type === 'result') {
            const result = event.data as unknown as RecommendationResponse;
            setRecommendations(result.recommendations);
            setIndicatorRelationships(result.indicator_relationships || []);
            setRecommendationSummary(result.summary || null);
            toast({
              title: `Found ${result.recommendations.length} recommendations`,
              description: `Reviewed ${result.total_evidence_reviewed} evidence records`,
              status: 'success',
            });
          } else if (event.type === 'error') {
            toast({ title: event.message || 'Recommendation failed', status: 'error' });
          }
        },
      );
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Recommendation failed';
      toast({ title: message, status: 'error' });
    } finally {
      setIsStreaming(false);
      setStreamStatus('');
    }
  };

  const toggleIndicator = useCallback(
    (indicator: IndicatorRecommendation) => {
      if (selectedIds.has(indicator.indicator_id)) {
        removeSelectedIndicator(indicator.indicator_id);
      } else {
        addSelectedIndicator(indicator);
      }
    },
    [selectedIds, addSelectedIndicator, removeSelectedIndicator],
  );

  return (
    <PageShell isLoading={kbLoading} loadingText="Loading knowledge base...">
      <PageHeader title="Indicator Recommendation" />

      {/* Knowledge Base Status */}
      {kbSummary && (
        <Alert status={kbSummary.loaded ? 'success' : 'warning'} mb={6}>
          <AlertIcon />
          Knowledge Base: {kbSummary.total_evidence} evidence records,{' '}
          {kbSummary.indicators_with_evidence} indicators with evidence
        </Alert>
      )}

      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
        {/* Left: Query Form */}
        <VStack spacing={6} align="stretch">
          <Card>
            <CardHeader>
              <Heading size="md">Project Context</Heading>
            </CardHeader>
            <CardBody>
              <VStack spacing={3} align="stretch">
                <Box>
                  <Text fontSize="sm" fontWeight="bold" color="gray.500">Project</Text>
                  <Text>{activeProject?.project_name || 'No project selected'}</Text>
                </Box>
                {activeProject?.design_brief && (
                  <Box>
                    <Text fontSize="sm" fontWeight="bold" color="gray.500">Design Brief</Text>
                    <Text fontSize="sm" color="gray.600">{activeProject.design_brief}</Text>
                  </Box>
                )}
                {activeProject?.koppen_zone_id && (
                  <HStack spacing={2} flexWrap="wrap">
                    <Badge>{activeProject.koppen_zone_id}</Badge>
                    {activeProject.space_type_id && <Badge>{activeProject.space_type_id}</Badge>}
                    {activeProject.lcz_type_id && <Badge>{activeProject.lcz_type_id}</Badge>}
                  </HStack>
                )}
              </VStack>
            </CardBody>
          </Card>

          <Card>
            <CardHeader pb={2}>
              <HStack justify="space-between">
                <VStack align="start" spacing={0}>
                  <Text fontSize="sm" fontWeight="bold" color="gray.500">Performance Dimensions</Text>
                  <Wrap spacing={1} mt={1}>
                    {selectedDimensions.map(id => {
                      const dim = DIMENSIONS.find(d => d.id === id);
                      return dim ? (
                        <WrapItem key={id}><Badge colorScheme="blue" fontSize="xs">{dim.name}</Badge></WrapItem>
                      ) : null;
                    })}
                    {selectedDimensions.length === 0 && <Text fontSize="xs" color="red.500">None selected</Text>}
                  </Wrap>
                </VStack>
                <Button size="xs" variant="ghost" onClick={toggleAdvanced} rightIcon={advancedOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}>
                  {advancedOpen ? 'Hide' : 'Customize'}
                </Button>
              </HStack>
            </CardHeader>
            <Collapse in={advancedOpen} animateOpacity>
              <CardBody pt={0}>
                <Text fontSize="xs" color="gray.500" mb={2}>
                  Pre-filled from project settings. Adjust if you want to narrow the recommendation scope.
                </Text>
                <CheckboxGroup
                  value={selectedDimensions}
                  onChange={(v) => setSelectedDimensions(v as string[])}
                >
                  <VStack align="stretch" spacing={2}>
                    {DIMENSIONS.map((dim) => (
                      <Checkbox key={dim.id} value={dim.id} size="sm">
                        {dim.name}
                      </Checkbox>
                    ))}
                  </VStack>
                </CheckboxGroup>
              </CardBody>
            </Collapse>
          </Card>

          <Button
            colorScheme="blue"
            size="lg"
            onClick={handleRecommend}
            isLoading={isStreaming}
            loadingText={streamStatus || 'Generating…'}
            isDisabled={!activeProject?.project_name || selectedDimensions.length === 0}
          >
            Get Recommendations
          </Button>
        </VStack>

        {/* Right: Results */}
        <VStack spacing={6} align="stretch">
          {/* Streaming output */}
          {isStreaming && (
            <Card borderColor="blue.200" borderWidth={1}>
              <CardHeader pb={2}>
                <HStack>
                  <Spinner size="sm" color="blue.400" />
                  <Text fontSize="sm" fontWeight="bold" color="blue.600">{streamStatus || 'Generating…'}</Text>
                </HStack>
              </CardHeader>
              {streamingText && (
                <CardBody pt={0}>
                  <Box
                    ref={streamBoxRef}
                    maxH="300px"
                    overflowY="auto"
                    bg="gray.50"
                    p={3}
                    borderRadius="md"
                    fontSize="xs"
                    fontFamily="mono"
                    whiteSpace="pre-wrap"
                    wordBreak="break-word"
                  >
                    {streamingText}
                  </Box>
                </CardBody>
              )}
            </Card>
          )}
          {/* Selected Indicators */}
          {selectedIndicators.length > 0 && (
            <Card>
              <CardHeader>
                <HStack justify="space-between">
                  <Heading size="md">Selected Indicators ({selectedIndicators.length})</Heading>
                  <Button size="xs" colorScheme="red" variant="ghost" onClick={clearSelectedIndicators}>
                    Clear All
                  </Button>
                </HStack>
              </CardHeader>
              <CardBody>
                <Wrap>
                  {selectedIndicators.map((ind) => (
                    <WrapItem key={ind.indicator_id}>
                      <Tag
                        size="lg"
                        colorScheme="green"
                        cursor="pointer"
                        onClick={() => removeSelectedIndicator(ind.indicator_id)}
                      >
                        <TagLabel>{ind.indicator_id}</TagLabel>
                      </Tag>
                    </WrapItem>
                  ))}
                </Wrap>
              </CardBody>
            </Card>
          )}

          {/* Recommendations */}
          {recommendations.length > 0 ? (
            <Card>
              <CardHeader>
                <Heading size="md">Recommendations</Heading>
              </CardHeader>
              <CardBody p={0}>
                <Accordion allowMultiple>
                  {recommendations.map((rec) => (
                    <RecommendationItem
                      key={rec.indicator_id}
                      rec={rec}
                      selected={selectedIds.has(rec.indicator_id)}
                      onToggle={toggleIndicator}
                    />
                  ))}
                </Accordion>
              </CardBody>
            </Card>
          ) : (
            <EmptyState
              icon={Lightbulb}
              title="No recommendations yet"
              description="Enter project details and select dimensions to get AI-powered indicator recommendations."
            />
          )}

          {/* Indicator Relationships */}
          {indicatorRelationships.length > 0 && (
            <Card>
              <CardHeader>
                <Heading size="sm">Indicator Relationships</Heading>
              </CardHeader>
              <CardBody pt={0}>
                <VStack align="stretch" spacing={2}>
                  {indicatorRelationships.map((rel, i) => (
                    <HStack key={i} fontSize="sm" spacing={2}>
                      <Badge colorScheme="blue">{rel.indicator_a}</Badge>
                      <Badge
                        colorScheme={rel.relationship_type === 'synergistic' ? 'green' : rel.relationship_type === 'inverse' ? 'red' : 'gray'}
                      >
                        {rel.relationship_type}
                      </Badge>
                      <Badge colorScheme="blue">{rel.indicator_b}</Badge>
                      {rel.explanation && (
                        <Text fontSize="xs" color="gray.500" noOfLines={1} flex={1}>{rel.explanation}</Text>
                      )}
                    </HStack>
                  ))}
                </VStack>
              </CardBody>
            </Card>
          )}

          {/* Recommendation Summary */}
          {recommendationSummary && (
            <Card>
              <CardHeader>
                <Heading size="sm">Summary</Heading>
              </CardHeader>
              <CardBody pt={0}>
                <VStack align="stretch" spacing={3}>
                  {recommendationSummary.key_findings.length > 0 && (
                    <Box>
                      <Text fontSize="sm" fontWeight="bold" mb={1}>Key Findings</Text>
                      <List spacing={1}>
                        {recommendationSummary.key_findings.map((f, i) => (
                          <ListItem key={i} fontSize="sm" display="flex" alignItems="start">
                            <Box as="span" color="green.500" mr={2} mt={0.5} flexShrink={0}>&#x2713;</Box>
                            {f}
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                  {recommendationSummary.evidence_gaps.length > 0 && (
                    <Box>
                      <Divider mb={2} />
                      <Text fontSize="sm" fontWeight="bold" mb={1}>Evidence Gaps</Text>
                      <List spacing={1}>
                        {recommendationSummary.evidence_gaps.map((g, i) => (
                          <ListItem key={i} fontSize="sm" display="flex" alignItems="start">
                            <Box as="span" color="orange.500" mr={2} mt={0.5} flexShrink={0}>&#x26A0;</Box>
                            {g}
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </VStack>
              </CardBody>
            </Card>
          )}
        </VStack>
      </SimpleGrid>

      {/* Navigation buttons for pipeline mode */}
      {routeProjectId && (
        <HStack justify="space-between" mt={6}>
          <Button as={Link} to={`/projects/${routeProjectId}/vision`} variant="outline">
            Back: Vision
          </Button>
          <Button as={Link} to={`/projects/${routeProjectId}/analysis`} colorScheme="blue">
            Next: Analysis
          </Button>
        </HStack>
      )}
    </PageShell>
  );
}

export default Indicators;
