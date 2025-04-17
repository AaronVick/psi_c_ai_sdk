#!/usr/bin/env python3
"""
Tests for the Hierarchical Knowledge Manager implementation.

These tests verify that the HierarchicalKnowledgeManager correctly manages
concept hierarchies, taxonomic branches, and ontological relationships.
"""

import unittest
import networkx as nx
from datetime import datetime

from psi_c_ai_sdk.schema import SchemaGraph
from psi_c_ai_sdk.knowledge.hierarchical_manager import HierarchicalKnowledgeManager


class TestHierarchicalKnowledgeManager(unittest.TestCase):
    """Test cases for the HierarchicalKnowledgeManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a schema graph
        self.schema_graph = SchemaGraph()
        
        # Initialize the manager
        self.manager = HierarchicalKnowledgeManager(schema_graph=self.schema_graph)
        
        # Create a test concept hierarchy
        self.create_test_hierarchy()
        
    def create_test_hierarchy(self):
        """Create a test concept hierarchy for testing."""
        # Create root concept
        self.schema_graph.add_node(
            "concept_root",
            node_type="concept",
            label="Root Concept",
            metadata={"created_at": datetime.now().isoformat()}
        )
        
        # Create first level concepts
        for i in range(3):
            concept_id = f"concept_level1_{i}"
            self.schema_graph.add_node(
                concept_id,
                node_type="concept",
                label=f"Level 1 Concept {i}",
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            # Connect to root
            self.schema_graph.add_edge(
                "concept_root",
                concept_id,
                edge_type="parent_child",
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            # Create second level concepts
            for j in range(2):
                sub_concept_id = f"concept_level2_{i}_{j}"
                self.schema_graph.add_node(
                    sub_concept_id,
                    node_type="concept",
                    label=f"Level 2 Concept {i}.{j}",
                    metadata={"created_at": datetime.now().isoformat()}
                )
                
                # Connect to parent
                self.schema_graph.add_edge(
                    concept_id,
                    sub_concept_id,
                    edge_type="parent_child",
                    metadata={"created_at": datetime.now().isoformat()}
                )
    
    def test_get_concept_hierarchy(self):
        """Test getting the concept hierarchy."""
        hierarchy = self.manager.get_concept_hierarchy()
        
        # Verify hierarchy structure
        self.assertIn("concepts", hierarchy)
        self.assertEqual(len(hierarchy["concepts"]), 10)  # 1 root + 3 level 1 + 6 level 2
        
        # Check that concepts have expected attributes
        for concept in hierarchy["concepts"]:
            self.assertIn("id", concept)
            self.assertIn("label", concept)
            self.assertIn("parents", concept)
            self.assertIn("children", concept)
    
    def test_create_taxonomic_branch(self):
        """Test creating a new taxonomic branch."""
        # Create a new branch
        branch_id = self.manager.create_taxonomic_branch(
            parent_concept="Root Concept",
            child_concepts=["New Child 1", "New Child 2", "New Child 3"],
            branch_name="test_branch"
        )
        
        # Verify branch creation
        self.assertEqual(branch_id, "test_branch")
        
        # Get updated hierarchy
        hierarchy = self.manager.get_concept_hierarchy(refresh=True)
        
        # Find the root concept
        root_concept = None
        for concept in hierarchy["concepts"]:
            if concept["label"] == "Root Concept":
                root_concept = concept
                break
        
        self.assertIsNotNone(root_concept)
        
        # Verify new children are connected
        child_labels = set()
        for child_id in root_concept["children"]:
            for concept in hierarchy["concepts"]:
                if concept["id"] == child_id:
                    child_labels.add(concept["label"])
        
        self.assertIn("New Child 1", child_labels)
        self.assertIn("New Child 2", child_labels)
        self.assertIn("New Child 3", child_labels)
    
    def test_get_lineage(self):
        """Test getting a concept's lineage."""
        # Get lineage for a level 1 concept
        lineage = self.manager.get_lineage("concept_level1_0")
        
        # Verify lineage structure
        self.assertIn("concept", lineage)
        self.assertIn("ancestors", lineage)
        self.assertIn("descendants", lineage)
        
        # Should have 1 ancestor (root) and 2 descendants (level 2)
        self.assertEqual(len(lineage["ancestors"]), 1)
        self.assertEqual(len(lineage["descendants"]), 2)
        
        # Verify ancestor is root
        self.assertEqual(lineage["ancestors"][0]["id"], "concept_root")
        
        # Verify descendants are level 2 concepts
        descendant_ids = [d["id"] for d in lineage["descendants"]]
        self.assertIn("concept_level2_0_0", descendant_ids)
        self.assertIn("concept_level2_0_1", descendant_ids)
    
    def test_find_common_ancestor(self):
        """Test finding common ancestor of multiple concepts."""
        # Find common ancestor of two level 2 concepts under the same parent
        common_ancestor = self.manager.find_common_ancestor([
            "concept_level2_0_0", 
            "concept_level2_0_1"
        ])
        
        # Common ancestor should be their parent
        self.assertEqual(common_ancestor, "concept_level1_0")
        
        # Find common ancestor of two level 2 concepts under different parents
        common_ancestor = self.manager.find_common_ancestor([
            "concept_level2_0_0", 
            "concept_level2_1_0"
        ])
        
        # Common ancestor should be the root
        self.assertEqual(common_ancestor, "concept_root")
        
        # Find common ancestor of unrelated concepts
        self.manager.create_taxonomic_branch(
            parent_concept="Unrelated Root",
            child_concepts=["Unrelated Child"],
            branch_name="unrelated_branch"
        )
        
        # Find all concept IDs with "Unrelated" in label
        unrelated_id = None
        hierarchy = self.manager.get_concept_hierarchy(refresh=True)
        for concept in hierarchy["concepts"]:
            if "Unrelated Child" in concept["label"]:
                unrelated_id = concept["id"]
                break
        
        common_ancestor = self.manager.find_common_ancestor([
            "concept_level2_0_0", 
            unrelated_id
        ])
        
        # Should have no common ancestor
        self.assertIsNone(common_ancestor)
    
    def test_merge_branches(self):
        """Test merging two taxonomic branches."""
        # Create two branches
        branch1 = self.manager.create_taxonomic_branch(
            parent_concept="Branch Parent 1",
            child_concepts=["Child 1A", "Child 1B", "Child 1C"],
            branch_name="branch1"
        )
        
        branch2 = self.manager.create_taxonomic_branch(
            parent_concept="Branch Parent 2",
            child_concepts=["Child 2A", "Child 2B"],
            branch_name="branch2"
        )
        
        # Merge branches
        merge_results = self.manager.merge_branches(
            source_branch="branch1",
            target_branch="branch2",
            conflict_resolution="target_wins"
        )
        
        # Verify merge results
        self.assertEqual(merge_results["source_branch"], "branch1")
        self.assertEqual(merge_results["target_branch"], "branch2")
        self.assertGreater(merge_results["merged_edges"], 0)
        
        # Get updated hierarchy
        hierarchy = self.manager.get_concept_hierarchy(refresh=True)
        
        # Verify branch1 children now have branch2 metadata
        branch1_parent_id = None
        for concept in hierarchy["concepts"]:
            if concept["label"] == "Branch Parent 1":
                branch1_parent_id = concept["id"]
                break
        
        # Check edges
        edge_found = False
        for edge in self.schema_graph.graph.edges(data=True):
            source, target, data = edge
            if source == branch1_parent_id:
                branch_id = data.get("metadata", {}).get("branch_id")
                if branch_id == "branch2":
                    edge_found = True
                    break
        
        self.assertTrue(edge_found)
    
    def test_get_concept_impact(self):
        """Test calculating concept impact."""
        # Get impact of root concept
        impact = self.manager.get_concept_impact("concept_root")
        
        # Verify impact metrics
        self.assertIn("direct_connections", impact)
        self.assertIn("hierarchy_influence", impact)
        self.assertIn("overall_impact", impact)
        
        # Root should have high hierarchy influence
        self.assertGreater(impact["hierarchy_influence"], 0.5)
        
        # Get impact of leaf concept
        leaf_impact = self.manager.get_concept_impact("concept_level2_0_0")
        
        # Leaf should have lower hierarchy influence
        self.assertLess(leaf_impact["hierarchy_influence"], impact["hierarchy_influence"])
        
        # Overall impact of root should be higher
        self.assertGreater(impact["overall_impact"], leaf_impact["overall_impact"])


if __name__ == "__main__":
    unittest.main() 